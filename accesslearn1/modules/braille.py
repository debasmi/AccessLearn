# modules/braille.py  — Grade-1 Unicode Braille converter

BASE = 0x2800
LETTER_MASKS = [
    1, 3, 9, 25, 17, 11, 27, 19, 10, 26, 5, 7, 13, 29, 21, 15,
    31, 23, 14, 30, 69, 71, 90, 77, 93, 85
]
NUMBER_SIGN  = chr(BASE + 60)   # dots 3-4-5-6
CAPITAL_SIGN = chr(BASE + 32)   # dot 6
SPACE        = chr(BASE + 0)

def letter_to_braille(ch: str) -> str:
    idx = ord(ch.lower()) - ord('a')
    if 0 <= idx < 26:
        return chr(BASE + LETTER_MASKS[idx])
    return '?'

def text_to_braille(text: str) -> str:
    """Convert plain English text to Unicode Braille (Grade 1)."""
    out = []
    i   = 0
    n   = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            out.append(SPACE); i += 1; continue
        if ch.isdigit():
            out.append(NUMBER_SIGN)
            while i < n and text[i].isdigit():
                d = text[i]
                out.append(letter_to_braille('j' if d == '0'
                            else chr(ord('a') + int(d) - 1)))
                i += 1
            continue
        if ch.isalpha():
            if ch.isupper():
                out.append(CAPITAL_SIGN)
            out.append(letter_to_braille(ch))
            i += 1; continue
        out.append('?'); i += 1
    return ''.join(out)
