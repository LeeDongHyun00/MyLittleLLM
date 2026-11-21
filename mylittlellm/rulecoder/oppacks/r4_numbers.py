"""Round 4 operators for numbers: counting formulas, relations between quadratic roots, named problems, grids.

Every operator is gated and every one but the grid builder is a definition or formula, so each is tried
only when the description names its concept. A numeric example is one small value; an operator open on
every task would reproduce many of them by accident. The search function of each operator is compiled
from its own template (see `formula`), so the program the search verifies is the program that is shipped.
"""
import functools
import math

# ---------------------------------------------------------------- helper sources
# Pasted into generated answers, so they use builtins, `math` and `functools` only. Names carry the pack
# suffix because helpers of different packs end up side by side in one generated answer. No call to the
# builtin sum(): a task may name its own function `sum`, which would shadow it inside the helper.

# The m-th positive non-multiple of k is m + (m - 1) // (k - 1): every k - 1 of them skip one multiple.
# Counting the non-multiples below `start` first turns "from start" into "from 1".
_NTH_NON_MULTIPLE = '''def _nth_non_multiple_r4n(k, n, start):
    m = (start - 1) - (start - 1) // k + n
    return m + (m - 1) // (k - 1)
'''

# Upright triangles of side k fit in T(n - k + 1) places of a triangle of side n, inverted ones in
# T(n - 2k + 1) (T is the triangular number); -1 is the usual answer when not even one fits.
_TRIANGLES_IN_TRIANGLE = '''def _triangles_in_triangle_r4n(n, k):
    if n < k:
        return -1
    upright = (n - k + 1) * (n - k + 2) // 2
    inverted = (n - 2 * k + 1) * (n - 2 * k + 2) // 2 if n >= 2 * k else 0
    return upright + inverted
'''

# The Bytelandian exchange: a coin n is worth the better of n itself and its parts n // 2, n // 3, n // 4,
# each valued the same way. Memoised, because the parts of different branches repeat.
_COIN_SPLIT = '''@functools.lru_cache(maxsize=None)
def _coin_split_value_r4n(n):
    if n <= 0:
        return 0
    return max(n, _coin_split_value_r4n(n // 2) + _coin_split_value_r4n(n // 3) + _coin_split_value_r4n(n // 4))
'''

# Taking the largest number made of nines that does not exceed n as one part leaves no carries to lose,
# which is what maximises the digit sum of the two parts (checked by brute force up to 3000).
_SPLIT_DIGIT_SUM = '''def _max_split_digit_sum_r4n(n):
    nines = 0
    while nines * 10 + 9 <= n:
        nines = nines * 10 + 9
    total, rest = 0, n - nines
    for part in (nines, rest):
        while part:
            total += part % 10
            part //= 10
    return total
'''


def register(lib):
    need, is_int, is_num = lib.need, lib.is_int, lib.is_num
    scope = {"math": math, "functools": functools}
    text_of = {}
    for source in (_NTH_NON_MULTIPLE, _TRIANGLES_IN_TRIANGLE, _COIN_SPLIT, _SPLIT_DIGIT_SUM):
        function, text = lib.helper(source)
        scope[function.__name__] = function
        text_of[function.__name__] = text

    def formula(arity, name, template, keywords, guard, helper=None, predicate=False, kind="knowledge"):
        """Register a gated operator whose search function is its own template behind a guard.

        The guard says where the operator applies and bounds the work; it never changes a value.
        """
        names = [f"a{i}" for i in range(arity)]
        body = eval(f"lambda {', '.join(names)}: " + template.format(*names), scope)

        def run(*args):
            guard(*args)
            value = body(*args)
            # A predicate must give a real bool: 0/1 would never equal a True/False target anyway.
            need(not predicate or value is True or value is False)
            return value
        {1: lib.unary, 2: lib.binary, 3: lib.ternary}[arity](
            name, run, template, keywords, gated=True, kind=kind, helper=text_of[helper] if helper else None)

    def ints(*ranges):
        """Guard: every argument is an int inside its (low, high) range."""
        def check(*args):
            need(all(is_int(a) and low <= a <= high for a, (low, high) in zip(args, ranges)))
        return check

    def ordered(low, high):
        """Guard: two ints with low <= a <= b <= high, an interval or a (small, large) pair."""
        def check(a, b):
            need(is_int(a) and is_int(b) and low <= a <= b <= high)
        return check

    def coefficients(a, b, c):
        # A quadratic needs a leading coefficient; the bound keeps products exact in floats as well.
        need(all(is_num(v) and abs(v) <= 10 ** 6 for v in (a, b, c)) and a != 0)

    # ------------------------------------------------------------ counting in intervals and grids
    # A number has an odd number of divisors exactly when it is a perfect square, so "numbers with odd
    # factors" and "perfect squares" between a and b are the same count.
    formula(2, "count_squares_between_r4n", "math.isqrt({1}) - math.isqrt({0} - 1)",
            "with odd factors|odd number of factors|odd number of divisors|number of perfect squares|"
            "count&perfect squares|perfect squares&between|perfect squares&range|perfect squares&interval|"
            "square numbers&between|square numbers&range",
            ordered(1, 10 ** 15))
    # A cube of side k can start at n - k + 1 places along each edge of a cube of side n.
    formula(2, "cubes_in_cube_r4n", "({0} - {1} + 1) ** 3",
            "cubes of size|cubes of side|cubes&in a cube|cubes&inside a cube|cubes&within a cube|subcube|sub-cube",
            lambda n, k: need(is_int(n) and is_int(k) and 1 <= k <= n <= 10 ** 6))
    formula(2, "equilateral_triangles_in_r4n", "_triangles_in_triangle_r4n({0}, {1})",
            "number of equilateral triangles|count&equilateral triangles|equilateral triangles&within|"
            "equilateral triangles&inside|equilateral triangles&formed",
            ints((1, 10 ** 6), (1, 10 ** 6)), "_triangles_in_triangle_r4n")
    # "From start" because tasks give the sequence as a range; the upper end of that range never
    # matters once the n-th number is reached.
    formula(3, "nth_non_multiple_from_r4n", "_nth_non_multiple_r4n({0}, {1}, {2})",
            "|".join(f"{nth}&{other}" for nth in ("nth", "n-th", "n'th", "kth", "k-th")
                     for other in ("not a multiple", "not multiple", "not divisible", "non-multiple")),
            ints((2, 10 ** 9), (1, 10 ** 9), (1, 10 ** 12)), "_nth_non_multiple_r4n")

    # ------------------------------------------------------------ optimisation problems with a closed form
    # The product of three non-negative integers with a fixed sum is largest when they differ by at most 1.
    formula(1, "max_cuboid_volume_r4n", "({0} // 3) ** (3 - {0} % 3) * ({0} // 3 + 1) ** ({0} % 3)",
            "maximum volume&sum|max volume&sum|largest volume&sum|greatest volume&sum|volume&sum of sides|"
            "volume&sum of the sides", ints((0, 10 ** 6)))
    formula(1, "max_split_digit_sum_r4n", "_max_split_digit_sum_r4n({0})", "two parts&digit|2 parts&digit",
            ints((0, 10 ** 18)), "_max_split_digit_sum_r4n")
    formula(1, "coin_split_value_r4n", "_coin_split_value_r4n({0})",
            "three parts&recursiv|3 parts&recursiv|bytelandian", ints((0, 10 ** 12)), "_coin_split_value_r4n")
    # Going from k to k + 1 flips the trailing ones of k and one more bit, so the total over 0..n is
    # n plus the exponent of 2 in n!, which Legendre's formula gives as 2n - popcount(n).
    formula(1, "consecutive_hamming_total_r4n", "2 * {0} - bin({0}).count('1')", "hamming&consecutive",
            ints((0, 10 ** 18)))
    # On an interval of two or more integers, 2 divides at least as many of them as any larger divisor.
    # The divisors of a single number all occur once; these tasks then answer the number itself.
    formula(2, "most_frequent_divisor_r4n", "{1} if {0} == {1} else 2", "occur&divisor|frequen&divisor",
            ordered(2, 10 ** 12))

    # ------------------------------------------------------------ roots of a x^2 + b x + c
    # Vieta: roots r and 2r give -b / a = 3r and c / a = 2r^2, so 2 b^2 = 9 a c; reciprocal roots have
    # product c / a = 1. These answer True/False; the core yes/no wording comes from another operator.
    formula(3, "quadratic_root_double_r4n", "2 * {1} * {1} == 9 * {0} * {2}",
            "quadratic&twice|quadratic&double|roots&twice|root&twice", coefficients, predicate=True)
    formula(3, "quadratic_roots_reciprocal_r4n", "{0} == {2}", "quadratic&reciprocal|root&reciprocal",
            coefficients, predicate=True)

    # ------------------------------------------------------------ named sequences and tables
    # Odd numbers up to n in order, then the even ones: the arrangement that "odd first" tasks index into.
    formula(1, "odd_then_even_upto_r4n", "list(range(1, {0} + 1, 2)) + list(range(2, {0} + 1, 2))",
            "odd elements first|odd numbers first|odd integers first|odds first|odd first", ints((1, 200)))
    # The twelve-year cycle with 4 AD as a year of the Rat; English names as most references give them.
    formula(1, "chinese_zodiac_r4n",
            "('Rat', 'Ox', 'Tiger', 'Rabbit', 'Dragon', 'Snake', 'Horse', 'Goat', 'Monkey', 'Rooster', 'Dog',"
            " 'Pig')[({0} - 4) % 12]", "chinese zodiac|zodiac&chinese|zodiac&year", ints((-10 ** 6, 10 ** 6)))
    # Cell (i, j) holds i * j, rows and columns counted from 0: the multiplication table of the indices.
    formula(2, "index_product_grid_r4n", "[[i * j for j in range({1})] for i in range({0})]",
            "two-dimensional array|two dimensional array|2d array|2-d array|2d list|2-d list",
            ints((1, 30), (1, 30)))

    # A rows x cols grid holding one value everywhere, the value usually quoted in the description.
    # Generic, not knowledge: it builds a shape and states no fact.
    def fill(rows, cols, value):
        need(is_int(rows) and is_int(cols) and 1 <= rows <= 30 and 1 <= cols <= 30)
        need(is_num(value) or value is None or isinstance(value, bool)
             or (isinstance(value, str) and len(value) <= 10))
    formula(3, "filled_grid_r4n", "[[{2}] * {1} for _ in range({0})]",
            "3d array|3-d array|2d array|2-d array|three-dimensional|two-dimensional|three dimensional|"
            "two dimensional|filled with|each element as", fill, kind="generic")

    # ------------------------------------------------------------ more words for existing operators
    def also_named(key, keywords):
        # The operator may belong to another pack; when that pack is not loaded the words name nothing,
        # and a missing operator must not stop the whole library from loading.
        try:
            lib.also_named(key, keywords)
        except KeyError:
            pass

    # A 3-D array is whole copies of one 2-D layer; the core `repeat` would concatenate the layers instead.
    also_named("seq_repeat_whole/2", "3d array|3-d array|three-dimensional|three dimensional")
    # "The kth element" names the 1-based lookup outright; without the phrase a coincidental .index() of
    # the same list wins the tie on the longer keyword ("find").
    also_named("kth/2", "kth element|k'th element|k-th element")
