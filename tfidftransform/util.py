def to_lower(s: str, language=None) -> str:
    """
    A lowercase conversion method, especially for handling Turkish 'I' -> 'ı' & 'İ' -> 'i' conversions

    :param s: String to be lower cased.
    :param language: The language such as 'turkish', 'english' etc.
    :return: The string lower cased.
    """
    if language == 'turkish':
        s = s.replace('İ', 'i').replace('I', 'ı')
    return s.lower()
