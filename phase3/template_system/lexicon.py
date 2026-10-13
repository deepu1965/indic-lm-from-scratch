from typing import Dict

# ── Attribute Vocabulary ──
# Hindi Attributes with natural phrasing
ATTR_HI = {
    "height": {
        "noun": "ऊँचाई",
        "adj_greater": "अधिक", 
        "adj_less": "कम",
        "fact_greater": "से अधिक",
        "fact_less": "से कम",
        "fact_equal": "के बराबर",
        "unit": "सेमी"
    },
    "age": {
        "noun": "उम्र",
        "adj_greater": "अधिक",
        "adj_less": "कम",
        "fact_greater": "से अधिक",
        "fact_less": "से कम",
        "fact_equal": "के बराबर",
        "unit": "वर्ष"
    },
    "price": {
        "noun": "कीमत",
        "adj_greater": "अधिक",
        "adj_less": "कम",
        "fact_greater": "से अधिक",
        "fact_less": "से कम",
        "fact_equal": "के बराबर",
        "unit": "रुपये"
    },
    "quantity": {
        "noun": "मात्रा",
        "adj_greater": "अधिक",
        "adj_less": "कम",
        "fact_greater": "से अधिक",
        "fact_less": "से कम",
        "fact_equal": "के बराबर",
        "unit": "इकाई"
    }
}

ATTR_NE = {
    "height": {
        "noun": "उचाइ",
        "adj_greater": "बढी",
        "adj_less": "कम",
        "fact_greater": "भन्दा बढी",
        "fact_less": "भन्दा कम",
        "fact_equal": "बराबर",
        "unit": "सेमी"
    },
    "age": {
        "noun": "उमेर",
        "adj_greater": "बढी",
        "adj_less": "कम",
        "fact_greater": "भन्दा बढी",
        "fact_less": "भन्दा कम",
        "fact_equal": "बराबर",
        "unit": "वर्ष"
    },
    "price": {
        "noun": "मूल्य",
        "adj_greater": "बढी",
        "adj_less": "कम",
        "fact_greater": "भन्दा बढी",
        "fact_less": "भन्दा कम",
        "fact_equal": "बराबर",
        "unit": "रुपैयाँ"
    },
    "quantity": {
        "noun": "मात्रा",
        "adj_greater": "बढी",
        "adj_less": "कम",
        "fact_greater": "भन्दा बढी",
        "fact_less": "भन्दा कम",
        "fact_equal": "बराबर",
        "unit": "एकाइ"
    }
}

def get_lexicon(language: str) -> Dict[str, Dict]:
    if language == "hi":
        return {"attributes": ATTR_HI}
    elif language == "ne":
        return {"attributes": ATTR_NE}
    else:
        raise ValueError(f"Unknown language: {language}")
