# Find Longest Repeating Pattern
# Longest Common Prefix Using Kasai's Algorithm

def suffix_array_construction(s):
    n = len(s)
    suffixes = sorted((s[i:], i) for i in range(n))
    suffix_array = [suffix[1] for suffix in suffixes]
    return suffix_array


def get_lcp(s, suffix_array):
    n = len(s)
    rank = [0] * n
    lcp = [0] * n
    for i, sa in enumerate(suffix_array):
        rank[sa] = i

    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = suffix_array[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
    return lcp


def find_lrp(s):
    suffix_array = suffix_array_construction(s)
    lcp = get_lcp(s, suffix_array)
    max_lcp = max(lcp)
    if max_lcp == 0:
        return ""  # No repeating pattern
    index = lcp.index(max_lcp)
    return s[suffix_array[index]:suffix_array[index] + max_lcp]
