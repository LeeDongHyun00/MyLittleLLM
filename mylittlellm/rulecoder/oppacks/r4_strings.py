"""Round 4 operators for strings and predicates: textbook regex tests, word answers, letter arithmetic, dates.

Most string tasks the engine still fails are regex exercises that answer in words ("Found a match!").
The word belongs to the r3_strings wrappers; what was missing is the regex test itself, as a gated
predicate the wrapper can be put around. The rest are named definitions (alphabet positions, seasons,
inversions) and edits that need one more step than a composition can make.
"""
import itertools
import math
import re

# Every name ends in _r4s (helpers start with _ and end in _r4s): packs are written in parallel and a
# clash stops the loader. Multi-line rules are kept as source text because the same text is pasted into
# generated answers; helpers import nothing, the engine compiles them with `math` only.
SEASON = '''def _season_r4s(month, day):
    names = ('january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
             'october', 'november', 'december')
    number = month if isinstance(month, int) else names.index(month.strip().lower()) + 1
    # The seasons change on the usual equinox and solstice dates: 20 March, 21 June, 22 September, 21 December.
    date = (number, day)
    if date < (3, 20) or date >= (12, 21):
        return 'winter'
    if date < (6, 21):
        return 'spring'
    if date < (9, 22):
        return 'summer'
    return 'autumn'
'''

MONTH_NAMES = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
               "november", "december")
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def register(lib):
    need, is_int, is_seq = lib.need, lib.is_int, lib.is_seq
    scope = {"re": re, "math": math, "itertools": itertools}

    def add(arity, name, guard, template, keywords, *, predicate=False, knowledge=False, helper=None):
        """Register an operator whose search function is its own template, compiled once.

        The guard only decides which inputs the search tries; the value always comes from the template
        text, so the program verified on the example is the program that is shipped.
        """
        local = dict(scope)
        if helper:
            function, helper = lib.helper(helper)
            local[function.__name__] = function
        params = ("_x0", "_x1", "_x2")[:arity]
        run = eval("lambda " + ", ".join(params) + ": " + template.format(*params), local)

        def fn(*args):
            need(guard(*args))
            value = run(*args)
            # A predicate that gives 0/1 or None would never equal a True/False target.
            need(not predicate or value is True or value is False)
            return value
        kind = "knowledge" if knowledge or predicate else "generic"
        {1: lib.unary, 2: lib.binary, 3: lib.ternary}[arity](name, fn, template, keywords, gated=True, kind=kind,
                                                             helper=helper)

    def text(x, low=0, high=600):
        # The search drops values with a longer repr, so longer strings can only be original arguments.
        return isinstance(x, str) and low <= len(x) <= high

    def nonempty(x):
        return text(x, 1)

    # ------------------------------------------------------------ word answers
    # Like the r3_strings wrappers: the predicate comes from another rule, this only words its truth value.
    add(1, "equal_words_r4s", lambda x: isinstance(x, bool), "'Equal' if {0} else 'Not Equal'", "equal")

    # ------------------------------------------------------------ textbook regex tests
    # Each test is the regex the exercise names, read the way its reference solution reads it (anchored or
    # searched). They return bools for the word wrappers; gates quote the exercise wording, because a single
    # True/False example is reproduced by half of all predicates.
    add(1, "lower_underscore_lower_r4s", text, "re.fullmatch(r'[a-z]+_[a-z]+', {0}) is not None",
        "lower&underscore", predicate=True)
    add(1, "starts_with_word_r4s", text, r"re.match(r'\w', {0}) is not None",
        "word at the beginning|word at the start|starts with a word|start with a word|begins with a word"
        "|begin with a word", predicate=True)
    # "Optional punctuation" lets non-space characters follow the last word, as in "python.".
    add(1, "ends_with_word_punct_r4s", text, r"re.search(r'\w\S*$', {0}) is not None",
        "word at the end&punctuation|ends with a word&punctuation|end with a word&punctuation", predicate=True)
    # Capitals followed by small letters at the end of the string, which is how both exercises of this
    # wording in the bank define a match.
    add(1, "upper_then_lower_r4s", text, "re.search(r'[A-Z]+[a-z]+$', {0}) is not None",
        "upper case letter followed by lower|uppercase letter followed by lower|capital letter followed by lower"
        "|upper case letter followed by small|uppercase letter followed by small|capital letter followed by small",
        predicate=True)
    # The quantifier exercises always use the letters a and b, so the letters are part of the definition.
    add(1, "a_then_any_bs_r4s", text, "re.search(r'ab*', {0}) is not None",
        "a followed by zero or more b|a' followed by zero or more 'b", predicate=True)
    add(1, "a_then_some_bs_r4s", text, "re.search(r'ab+', {0}) is not None",
        "a followed by one or more b|a' followed by one or more 'b", predicate=True)
    add(1, "a_anything_b_end_r4s", text, "re.search(r'a.*b$', {0}) is not None",
        "a' followed by anything&ending in 'b|a followed by anything&ending in b"
        "|a' followed by anything&ends with 'b|a followed by anything&ends with b"
        "|a' followed by anything&ending with 'b|a followed by anything&ending with b", predicate=True)

    # ------------------------------------------------------------ words of a length given in words
    # "Five characters long" is a number word, which the engine does not read as a constant (a digit is
    # read, and the core `of_length` takes it). A word is a run of word characters, as `\b\w{k}\b` reads it.
    # The number must follow a word that introduces one length ("all five characters", "of five letters"):
    # in "three, four, five characters" or "at least four characters" it is not the length of every word.
    lead = ("all ", "find ", "of ", "with ", "exactly ", "are ", "is ", "having ")
    for k, word in NUMBER_WORDS.items():
        names = [f"{before}{w} {unit}" for w in (word, str(k)) for before in lead for unit in ("char", "letter")]
        names += [f"{w}-{unit}" for w in (word, str(k)) for unit in ("char", "letter")]
        add(1, f"words_of_{k}_chars_r4s", nonempty, "re.findall(r'\\b\\w{{" + str(k) + "}}\\b', {0})",
            "|".join(names + [f"length {word}", f"length of {word}"]))

    # ------------------------------------------------------------ letters as alphabet positions
    # a=1 ... z=26. Adding the letters of a word and turning the total back into a letter are two natural
    # steps; the second wraps around the alphabet, so a total of 27 is 'a' again.
    letters = lambda x: text(x, 1, 200) and x.isascii() and x.isalpha()  # noqa: E731
    add(1, "alpha_position_sum_r4s", letters, "sum(ord(c) - ord('a') + 1 for c in {0}.lower())",
        "adding all the characters|adding the characters|adding all the letters|adding the letters"
        "|sum&alphabet&position|add&alphabet&position|sum&alphabetical&position|add&alphabetical&position"
        "|alphabet value|alphabetical value|letter value|word value|alphabet score", knowledge=True)
    add(1, "alphabet_letter_r4s", lambda n: is_int(n) and 1 <= n <= 10 ** 6, "chr(ord('a') + ({0} - 1) % 26)",
        "character made by|letter made by|adding all the characters|adding the characters|adding all the letters"
        "|alphabet&position|letter of the alphabet|corresponding letter|corresponding character", knowledge=True)

    # ------------------------------------------------------------ dates
    # Tasks give the month by name or by number; both read the same calendar.
    def month(m):
        return (text(m, 3, 12) and m.strip().lower() in MONTH_NAMES) or (is_int(m) and 1 <= m <= 12)

    add(2, "season_r4s", lambda m, d: month(m) and is_int(d) and 1 <= d <= 31, "_season_r4s({0}, {1})", "season",
        knowledge=True, helper=SEASON)

    # ------------------------------------------------------------ sequences
    # Every inversion is a local (adjacent) one exactly when no element is smaller than the largest element
    # two or more places before it: a linear scan instead of counting pairs, which hidden tests may make slow.
    # Two items always have as many inversions of both kinds, so the search needs at least three.
    add(1, "only_local_inversions_r4s", lambda x: lib.numseq(x) and 3 <= len(x) <= 1000,
        "all(m <= v for m, v in zip(itertools.accumulate({0}, max), {0}[2:]))",
        "inversion&two types|inversion&same|global&inversion|local&inversion", predicate=True)

    # Elements present in both lists, read in each list's own order. A list and its own tuple copy are not
    # two lists: the search pairs every argument with its conversions.
    def two_seqs(a, b):
        return (is_seq(a) and is_seq(b) and 0 < len(a) <= 500 and 0 < len(b) <= 500
                and not (type(a) is not type(b) and list(a) == list(b)))

    # "Same elements in the same order" is plain equality, so the gate wants the common elements named.
    add(2, "same_relative_order_r4s", two_seqs, "[v for v in {0} if v in {1}] == [v for v in {1} if v in {0}]",
        "common&same order|common&in order|relative order", predicate=True)

    # ------------------------------------------------------------ digits
    # Places alternate from either end alike for the equality, so the string is read from the left. A
    # one-digit number has no second place to compare with, which its reference answers with False.
    add(1, "digit_place_products_equal_r4s", lambda n: is_int(n) and 0 <= n <= 10 ** 18,
        "{0} >= 10 and math.prod(int(d) for d in str({0})[::2]) == math.prod(int(d) for d in str({0})[1::2])",
        "product of digits&even and odd|product of digits&odd and even|product&digit&even&odd&place"
        "|product&digit&even&odd&position", predicate=True)

    # ------------------------------------------------------------ edits
    # Abbreviating a word keeps the capital of the word it replaces ("Road" becomes "Rd."). Both words come
    # from the description, so the gate is the verb.
    def abbreviation(x, word, short):
        return (text(x, 1) and text(word, 1, 30) and text(short, 1, 30) and word.lower() != short.lower()
                and word.lower() in x.lower())

    add(3, "abbreviate_r4s", abbreviation,
        "re.sub(r'(?<!\\w)' + re.escape({1}) + r'(?!\\w)', lambda m: {2}[:1].upper() + {2}[1:] "
        "if m.group(0)[:1].isupper() else {2}, {0}, flags=re.IGNORECASE)", "abbreviat")
    # "Spaces, commas, or dots" is a class the description spells out; the engine reads each of its characters
    # as a separate constant, so the class itself has to be the rule. The replacement is a constant such as
    # ":", and a backslash would make re.sub read it as an escape. The count limits how many are replaced.
    replacement = lambda c: text(c, 1, 3) and "\\" not in c  # noqa: E731
    add(2, "replace_space_comma_dot_r4s", lambda x, c: text(x, 1) and replacement(c), "re.sub(r'[ ,.]', {1}, {0})",
        "space&comma&dot")
    add(3, "replace_space_comma_dot_n_r4s",
        lambda x, c, n: text(x, 1) and replacement(c) and is_int(n) and 0 < n <= 1000,
        "re.sub(r'[ ,.]', {1}, {0}, count={2})", "space&comma&dot")
