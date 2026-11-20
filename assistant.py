"""Velmora help-center assistant: BM25 retrieval + extractive answers.

No API key, no model download, standard library only.
Design choices (see README.md):
  * sentence-level BM25, so the answer is the exact sentence(s) from the help center;
  * "I don't know" when the question's important words are not covered by the best article;
  * conflicting articles: the most recent wins if the older one is clearly outdated,
    otherwise both are shown and the inconsistency is stated.
Nothing in this file is copied from eval.jsonl.
"""
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

TODAY = date(2026, 7, 1)          # the case study says "pretend today is 1 July 2026"
STALE_AFTER_DAYS = 365            # an article this much older than a conflicting one is outdated

STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being
do does did can could should would will may might i me my we our you your it its this that these those
there here what which who whom how when where why not no than then so such just about into onto up out
have has had get got much many use used need tell please know same still buy bought purchase purchased something anything thing anymore into back t s""".split())
# NB: "no", "not" are dropped from matching, they are handled by the answer text itself.

# Small domain thesaurus: query word -> words that the help center uses for the same idea.
DURATION = ["within", "days", "business", "hours", "months", "minutes", "years"]
SYNONYMS = {
    "deadline": ["before"], "delivery": ["deliver", "delivery"], "deliver": ["delivery", "deliver"], "payment": ["pay", "payment"], "methods": ["pay", "payment"],
    "delete": ["deletion"], "unsubscribe": ["unsubscribe", "newsletter"],
    "package": ["parcel"], "packages": ["parcel"], "shipment": ["parcel"], "order": ["order"],
    "week": ["7", "days"], "stuck": ["moved", "tracking"],
    "profile": ["account", "password", "login"], "login": ["password", "account"],
    "remember": ["forgot", "forgotten"], "forgot": ["forgot"], "locked": ["password"],
    "cost": ["price", "costs", "eur"], "price": ["costs", "eur"], "fee": ["costs", "eur"],
    "pay": ["payment"], "paid": ["payment"],
    "mistake": ["return", "unused"], "refund": ["refund", "refunded"],
    "broken": ["damaged", "defective"], "damage": ["damaged"], "defect": ["defective"],
    "faulty": ["defective"], "ship": ["deliver", "delivery"], "shipping": ["delivery"],
    "send": ["return", "ship"],
    "hours": ["opening", "monday", "friday"], "open": ["monday", "friday"],
    "notified": ["notify"], "notify": ["notify"],
    "stock": ["stock"], "voucher": ["miles", "voucher"],
    "discount": ["discount", "off"], "students": ["student"], "country": ["countries", "outside"],
    "countries": ["countries", "outside"],
}
# Any country name used in a question means "is that country covered?" -> shipping-countries vocabulary.
COUNTRIES = set("""afghanistan albania algeria andorra angola argentina armenia australia austria azerbaijan
bangladesh belarus bolivia bosnia brazil bulgaria cambodia canada chile china colombia croatia cuba cyprus
czechia denmark ecuador egypt estonia ethiopia finland ghana greece hungary iceland india indonesia iran iraq
ireland israel japan jordan kazakhstan kenya korea kuwait latvia lebanon libya lithuania luxembourg malaysia
malta mexico monaco morocco nepal norway pakistan peru philippines poland qatar romania russia serbia
singapore slovakia slovenia sweden switzerland syria taiwan thailand tunisia turkey ukraine uk usa uae
venezuela vietnam yemen britain england scotland wales america""".split())
COUNTRY_EXPANSION = ["countries", "outside", "european", "union", "ship", "deliver"]


DURATION_WORDS = {"long", "take", "takes", "fast", "quickly"}


UNIT = {"day": "time", "days": "time", "hour": "hour", "hours": "hour", "month": "month", "months": "month",
        "year": "year", "years": "year", "minute": "minute", "minutes": "minute", "eur": "eur", "%": "eur",
        "mile": "mile", "miles": "mile", "times": "times"}


def quantities(text: str):
    """{(value, unit)} found in a sentence, e.g. '14 days' -> ('14','time'); 'business/working days' too."""
    out = set()
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(?:-|\s)?\s*(%|(?:business\s+|working\s+)?[a-z]+)", text.lower()):
        raw_unit = m.group(2)
        unit = "bizday" if re.match(r"(business|working)\s+days", raw_unit) else raw_unit.split()[0]
        if unit == "bizday":
            out.add((m.group(1), "bizday"))
        elif unit in UNIT:
            out.add((m.group(1), UNIT[unit]))
    return out


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str):
    toks = re.findall(r"\d+(?:[.,:]\d+)?|[a-z]+", text.lower().replace("-", " "))
    return [stem(t) for t in toks if t not in STOPWORDS]


class Term(frozenset):
    """A query word with its alternatives; `own` = what the user really wrote (stemmed)."""
    own = frozenset()


def make_term(alts, own):
    t = Term(alts)
    t.own = frozenset(own)
    return t


@dataclass
class Sentence:
    article: str
    title: str
    updated: date
    text: str
    tokens: list


class Assistant:
    def __init__(self, kb_dir="kb", k1=1.4, b=0.6, min_coverage=0.45):
        self.k1, self.b, self.min_coverage = k1, b, min_coverage
        self.articles, self.sentences = {}, []
        self._load(Path(kb_dir))
        self._index()

    # ---------- indexing ----------
    def _load(self, kb_dir: Path):
        for f in sorted(kb_dir.glob("*.md")):
            raw = f.read_text(encoding="utf-8").replace("\r\n", "\n")
            _, head, body = raw.split("---", 2)
            meta = dict(l.split(": ", 1) for l in head.strip().splitlines())
            upd = date.fromisoformat(meta["updated"])
            body = "\n".join(l for l in body.splitlines() if not l.startswith("#")).strip()
            self.articles[meta["id"]] = {**meta, "updated": upd, "body": body}
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", body):
                s = s.strip()
                if s:
                    self.sentences.append(Sentence(meta["id"], meta["title"], upd, s, tokenize(s)))

    def _index(self):
        n = len(self.sentences)
        self.avgdl = sum(len(s.tokens) for s in self.sentences) / n
        df = {}
        for s in self.sentences:
            for t in set(s.tokens):
                df[t] = df.get(t, 0) + 1
        self.df, self.n = df, n
        self.max_idf = math.log(1 + (n + 0.5) / 0.5)
        self.title_tokens = {a: set(tokenize(m["title"])) for a, m in self.articles.items()}
        self.article_tokens = {a: set(tokenize(m["body"])) | self.title_tokens[a]
                               for a, m in self.articles.items()}

    def idf(self, t):
        d = self.df.get(t, 0)
        return math.log(1 + (self.n - d + 0.5) / (d + 0.5))

    # ---------- query processing ----------
    def analyse(self, question: str):
        """Return (list of query terms, each a set of alternatives) and the weights."""
        q = re.sub(r"cut[\s-]?off", "deadline", question.lower())
        raw = re.findall(r"[a-z]+|\d+", q)
        terms, duration_added = [], False
        for w in raw:
            if w in DURATION_WORDS:                      # "how long ... take" -> a single duration term
                if not duration_added:
                    terms.append(make_term({stem(d) for d in DURATION}, {stem(w) for w in DURATION_WORDS}))
                    duration_added = True
                continue
            if w in STOPWORDS:
                continue
            alts = {stem(w)}
            for s in SYNONYMS.get(w, []):
                alts.add(stem(s))
            if w in COUNTRIES:
                alts |= {stem(x) for x in COUNTRY_EXPANSION}
            terms.append(make_term(alts, {stem(w)}))
        return terms

    def _term_idf(self, alts):
        return max((self.idf(a) for a in alts if a in self.df), default=self.max_idf)

    def lex_score(self, s: Sentence, terms):
        """BM25 over the sentence itself."""
        tf = {}
        for t in s.tokens:
            tf[t] = tf.get(t, 0) + 1
        dl, score = len(s.tokens), 0.0
        for alts in terms:
            hit = [a for a in alts if a in tf]
            if hit:
                a = max(hit, key=lambda x: self.idf(x))
                f = tf[a]
                score += self.idf(a) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return score

    def title_bonus(self, s: Sentence, terms):
        tt, bonus = self.title_tokens[s.article], 0.0
        for alts in terms:
            if alts.own & tt:
                bonus += 1.0 * self._term_idf(alts)
            elif alts & tt:
                bonus += 0.3 * self._term_idf(alts)
        return bonus

    def context_credit(self, s: Sentence, terms):
        """Question words missing from this sentence but present elsewhere in its article count a little:
        an article that explains the whole question is a better source than one that matches one word."""
        toks, credit = set(s.tokens), 0.0
        for alts in terms:
            if not any(a in toks for a in alts) and any(a in self.article_tokens[s.article] for a in alts):
                credit += 0.35 * self._term_idf(alts)
        return credit

    def score_sentence(self, s: Sentence, terms):
        return self.lex_score(s, terms) + self.title_bonus(s, terms) + self.context_credit(s, terms)

    def _same_topic(self, a: Sentence, b: Sentence, terms):
        ha, hb = self.hits(a, terms), self.hits(b, terms)
        return len(ha & hb) >= 2 and len(ha & hb) / len(ha | hb) >= 0.6

    def _prefer_recent(self, top, ranked, terms, notes):
        """Same topic in a much newer article -> the older article is outdated, use the newer one."""
        best, top_lex = top, self.lex_score(top, terms)
        for _, s in ranked:
            if s.article != best.article and (s.updated - best.updated).days > STALE_AFTER_DAYS \
                    and self._same_topic(best, s, terms) and self.lex_score(s, terms) >= 0.6 * top_lex:
                notes.append(f"Ignored {best.article} ({best.title}): last updated {best.updated}, "
                             f"superseded by {s.article} (updated {s.updated}).")
                best = s
        return best

    def hits(self, s: Sentence, terms):
        """Indices of the query terms that the sentence matches itself (not through its title)."""
        toks = set(s.tokens)
        return {i for i, alts in enumerate(terms) if any(a in toks for a in alts)}

    def coverage(self, article_id, terms):
        """Share of the question's (idf-weighted) words that the article can account for."""
        tot = got = 0.0
        for alts in terms:
            w = self._term_idf(alts)
            tot += w
            if any(a in self.article_tokens[article_id] for a in alts):
                got += w
        return got / tot if tot else 0.0

    # ---------- answering ----------
    def ask(self, question: str) -> dict:
        terms = self.analyse(question)
        ranked = sorted(((self.score_sentence(s, terms), s) for s in self.sentences),
                        key=lambda x: -x[0])
        if not terms or ranked[0][0] <= 0:
            return self._unknown(question, 0.0, "no matching article")
        top_score, top = ranked[0]
        cov = self.coverage(top.article, terms)
        if cov < self.min_coverage:
            return self._unknown(question, cov, "the help center does not cover the question's key words")

        top = self._prefer_recent(top, ranked, terms, notes_out := [])
        top_score = self.score_sentence(top, terms)
        top_hits, top_q = self.hits(top, terms), quantities(top.text)
        top_lex = self.lex_score(top, terms)
        chosen = [top]
        for sc, s in ranked[1:]:
            if s in chosen:
                continue
            if s.article == top.article:
                if sc >= 0.25 * top_score and len(chosen) < 3:
                    chosen.append(s)
                continue
            if any(c.article == s.article for c in chosen):
                continue
            h, q = self.hits(s, terms), quantities(s.text)
            same_topic = len(top_hits) >= 2 and top_hits <= h
            same_kind = {u for _, u in q} & {u for _, u in top_q}
            differs = {v for v, u in q if u in {x for _, x in top_q}} - {v for v, _ in top_q}
            if same_topic and same_kind and differs and self.lex_score(s, terms) >= 0.7 * top_lex:
                chosen.append(s)
        chosen.sort(key=lambda s: (s.article != top.article, self.sentences.index(s)))

        # conflict handling between different articles
        notes, final = list(notes_out), []
        for s in chosen:
            clash = [o for o in final if o.article != s.article]
            outdated = False
            for o in clash:
                gap = (o.updated - s.updated).days
                if gap > STALE_AFTER_DAYS:           # s is much older than o
                    outdated = True
                    notes.append(f"Ignored {s.article} ({s.title}): last updated {s.updated}, "
                                 f"superseded by {o.article} (updated {o.updated}).")
            if not outdated:
                final.append(s)
        # an older article already in `final` while a much newer one arrives: drop the old one
        for s in list(final):
            for o in final:
                if o.article != s.article and (o.updated - s.updated).days > STALE_AFTER_DAYS:
                    final.remove(s)
                    notes.append(f"Ignored {s.article} ({s.title}): last updated {s.updated}, "
                                 f"superseded by {o.article} (updated {o.updated}).")
                    break

        articles = list(dict.fromkeys(s.article for s in final))
        if len(articles) > 1:
            lines = []
            for a in articles:
                txt = " ".join(x.text for x in final if x.article == a)
                lines.append(f"- {txt} (source: {self.articles[a]['title']}, {a})")
            answer = ("Our help center gives different information depending on the case:\n"
                      + "\n".join(lines)
                      + "\nIf you are unsure which one applies to your order, please contact support.")
        else:
            answer = " ".join(x.text for x in final)
        return {"question": question, "answerable": True, "answer": answer,
                "notes": list(dict.fromkeys(notes)),
                "sources": [{"id": a, "title": self.articles[a]["title"],
                             "updated": str(self.articles[a]["updated"])} for a in articles],
                "confidence": round(cov, 2)}

    def _unknown(self, question, cov, why):
        return {"question": question, "answerable": False,
                "answer": "I'm sorry, I couldn't find this in our help center, so I don't want to guess. "
                          "Please contact customer support (live chat or email, Monday to Friday, "
                          "9:00 to 18:00 CET) and keep your order number ready.",
                "sources": [], "confidence": round(cov, 2), "notes": [why]}
