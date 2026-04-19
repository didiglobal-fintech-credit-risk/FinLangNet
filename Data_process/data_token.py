"""Feature tokenizers for FinLangNet's Non-Sequential Module input.

The Non-Sequential Module (DeepFM) requires all static personal features
to be represented as integer indices.  This module provides:
  - A generic Tokenizer class that maps categorical values to integer indices.
  - Pre-fitted tokenizer instances for:
      * inquiry_type_tokenizer: encodes the type code of external credit inquiries
        (e.g., 'F'=finance, 'TC'=telecom, 'M'=mortgage, …).
      * month_income_tokenizer: encodes the declared monthly income bracket
        (denominated in MXN for the Mexico cash-loan product).

These tokenizers are used in sentences_load.py during the `process_one_sample`
function to convert raw categorical fields into embedding indices.
"""


class Tokenizer:
    """Simple vocabulary-based tokenizer for categorical feature encoding.

    Maps a fixed set of token strings to integer indices (and back).  Unknown
    tokens encountered at inference time are mapped to index 0 (<unk>).

    Attributes:
        token_to_index (dict): Forward mapping: token string → integer index.
        index_to_token (dict): Reverse mapping: integer index → token string.
        unk_token (str):  String used to represent unknown tokens.
        unk_index (int):  Integer index assigned to unknown tokens (default: 0).
    """

    def __init__(self, token_to_index: dict = None, index_to_token: dict = None):
        """Initialize the tokenizer, optionally with pre-built mappings.

        Args:
            token_to_index: Pre-built forward mapping. Built via fit() if None.
            index_to_token: Pre-built reverse mapping. Built via fit() if None.
        """
        self.token_to_index = token_to_index or {}
        self.index_to_token = index_to_token or {}
        self.unk_token       = "<unk>"
        self.unk_index       = 0

    def fit(self, data: list) -> None:
        """Build vocabulary mappings from a list of tokens.

        Tokens are sorted to ensure a deterministic, reproducible mapping
        across runs.  Indices start at 1; index 0 is reserved for <unk>.

        Args:
            data: Flat list of token strings defining the vocabulary.
        """
        unique_tokens       = sorted(set(data))
        self.token_to_index = {token: idx for idx, token in enumerate(unique_tokens, start=1)}
        self.index_to_token = {idx: token for token, idx in self.token_to_index.items()}
        self.unk_index      = 0

    def transform(self, data: list) -> list:
        """Convert a list of token strings to integer indices.

        Args:
            data: List of token strings.

        Returns:
            List of integer indices.  Unseen tokens map to unk_index (0).
        """
        return [self.token_to_index.get(item, self.unk_index) for item in data]

    def reverse_transform(self, data: list) -> list:
        """Convert a list of integer indices back to token strings.

        Args:
            data: List of integer indices.

        Returns:
            List of token strings.  Unknown indices map to unk_token ("<unk>").
        """
        return [self.index_to_token.get(index, self.unk_token) for index in data]


# ── Credit Inquiry Type Tokenizer ────────────────────────────────────────────
# Each inquiry record includes a type code indicating which third-party bureau
# or product category requested the credit check.
inquiry_type_list = [
    'F',  'TC', 'NC', 'M',  'PP', 'R',  'Q',  'LC', 'CA', 'OT',
    'AM', 'PG', 'E',  'PN', 'AE', 'P',  'PM', 'AR', 'NG', 'A',
    'FI', 'L',  'null', 'TG', 'FT', 'LR', 'CO', 'CF', 'EQ', 'VR',
    'ED', 'AV', 'MC', 'BL', 'SH', 'HE',
]
inquiry_type_tokenizer = Tokenizer()
inquiry_type_tokenizer.fit(inquiry_type_list)


# ── Monthly Income Bracket Tokenizer ─────────────────────────────────────────
# Declared monthly income is collected as a discrete bracket during the loan
# application process (currency: MXN – Mexican Pesos).
month_income_list = [
    '5,000-10,000MXN',
    '10,000-20,000MXN',
    '3,000-5,000MXN',
    '20,000-50,000MXN',
    '1,000-3,000MXN',
    'below 1,000MXN',
    'above 50,000MXN',
]
month_income_tokenizer = Tokenizer()
month_income_tokenizer.fit(month_income_list)
