TARGET_COLUMN = "label"
METADATA_COLUMNS = [
    "FILENAME",
    "URL",
    "Domain",
    "Title",
]
LEAKAGE_COLUMNS = [
    "URLSimilarityIndex",
    "URLTitleMatchScore",
    "DomainTitleMatchScore",
    "TLDLegitimateProb",
    "URLCharProb",
]
CONTENT_FEATURES = [
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
]
COLUMNS_TO_DROP = METADATA_COLUMNS + LEAKAGE_COLUMNS + CONTENT_FEATURES