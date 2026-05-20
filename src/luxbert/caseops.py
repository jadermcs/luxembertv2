"""CaseOps: a reversible case-folding pre-normalization for tokenizers.

Idea (as requested): instead of letting the tokenizer carry uppercase letters,
strip case from the character stream and re-encode it as an explicit *operation*.
Every uppercase letter ``X`` is rewritten as ``<MARKER>x`` — a marker token
followed by the lowercase letter.

Why this can help a low-resource encoder
----------------------------------------
"Lëtzebuerg", "lëtzebuerg" and "LËTZEBUERG" would otherwise occupy different BPE
pieces and different embeddings. After CaseOps they share the same lowercase
sub-words, and *casing* becomes a tiny, highly reusable signal (the marker).
That concentrates the limited LB data on fewer, better-estimated embeddings —
the same data-efficiency motivation as the rest of this project.

The transform is a bijection on text (modulo unicode case round-trip quirks,
which we measure rather than assume), so a model trained on CaseOps text can be
scored against the *original* byte stream for a fair bits-per-byte comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default marker: U+2191 "UPWARDS ARROW". Visible (easy to debug) and absent
# from normal Luxembourgish web text. Any literal occurrence in the source is
# escaped by doubling, so reversibility holds even if it does appear.
DEFAULT_MARKER = "↑"


@dataclass(frozen=True)
class CaseOps:
    """Reversible uppercase -> ``marker + lowercase`` re-encoding.

    Parameters
    ----------
    marker:
        Single character emitted before a lowercased uppercase letter.
    """

    marker: str = DEFAULT_MARKER

    def __post_init__(self) -> None:
        if len(self.marker) != 1:
            raise ValueError("marker must be exactly one character")

    def _is_invertible_upper(self, ch: str) -> bool:
        """True if ``ch`` is an uppercase letter we can fold and restore exactly."""
        low = ch.lower()
        # Must be genuinely cased (lower differs), stay a single char when
        # lowercased, and round-trip back to itself when re-uppercased.
        return ch != low and len(low) == 1 and low.upper() == ch

    def encode(self, text: str) -> str:
        """Rewrite uppercase letters as ``marker + lowercase``.

        Literal markers in the source are escaped as ``marker + marker``.
        """
        out: list[str] = []
        m = self.marker
        for ch in text:
            if ch == m:
                out.append(m)
                out.append(m)
            elif self._is_invertible_upper(ch):
                out.append(m)
                out.append(ch.lower())
            else:
                out.append(ch)
        return "".join(out)

    def decode(self, text: str) -> str:
        """Invert :meth:`encode`."""
        out: list[str] = []
        m = self.marker
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == m:
                nxt = text[i + 1] if i + 1 < n else ""
                if nxt == m:  # escaped literal marker
                    out.append(m)
                    i += 2
                elif nxt:  # marker + letter -> uppercase letter
                    out.append(nxt.upper())
                    i += 2
                else:  # dangling trailing marker -> keep literal
                    out.append(m)
                    i += 1
            else:
                out.append(ch)
                i += 1
        return "".join(out)

    def roundtrip_ok(self, text: str) -> bool:
        return self.decode(self.encode(text)) == text
