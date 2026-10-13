import hashlib
import random
from typing import List, Dict

HINDI_MALE = [
    "आरव", "अमन", "अमित", "अंकित", "आकाश", "अजय", "अभिषेक", "आदित्य", "अरुण", "अशोक",
    "दीपक", "दिनेश", "राहुल", "रोहित", "रवि", "राज", "राजेश", "राकेश", "सुरेश", "सुमित",
    "विवेक", "विनय", "मोहित", "मनोज", "नितिन", "पंकज", "प्रवीण", "करण", "कुणाल", "वरुण"
]
HINDI_FEMALE = [
    "आशा", "अनिता", "अंजली", "अदिति", "अमृता", "कविता", "किरण", "कोमल", "नेहा", "निधि",
    "निशा", "पूजा", "प्रिया", "राधा", "रीमा", "रुचि", "साक्षी", "सोनिया", "सुनीता", "स्वाति",
    "मेघा", "श्रेया", "तन्वी", "दिव्या", "पायल", "रिया", "सिमरन", "मुस्कान", "ज्योति", "मीनाक्षी"
]

NEPALI_MALE = [
    "आकाश", "अनिल", "अर्जुन", "अशोक", "अमित", "आदित्य", "आशिष", "बिनोद", "दीपक", "दिपेश",
    "दिनेश", "गणेश", "गोपाल", "हरि", "किरण", "कृष्ण", "मनोज", "मिलन", "नरेश", "निश्चल",
    "प्रकाश", "प्रदीप", "राजन", "रमेश", "रोशन", "सागर", "सन्तोष", "सुरेश", "विवेक", "विनोद"
]
NEPALI_FEMALE = [
    "अञ्जली", "अनिता", "आशा", "अस्मिता", "आयुषा", "कविता", "करिश्मा", "सुशीला", "कोपिला", "निशा",
    "नेहा", "निर्मला", "पूजा", "प्रिया", "राधा", "रमा", "रीमा", "रुचि", "सविना", "स्मृति",
    "सुष्मा", "मनिषा", "सीता", "गीता", "सुनिता", "मञ्जु", "सरस्वती", "लक्ष्मी", "पार्वती", "दुर्गा"
]

def _build_pool(males: List[str], females: List[str]) -> Dict[str, List[str]]:
    # Fixed deterministic split: 40 train, 10 val, 10 test.
    # Take 20M+20F for train, 5M+5F for val, 5M+5F for test
    train = males[:20] + females[:20]
    val = males[20:25] + females[20:25]
    test = males[25:30] + females[25:30]
    # Sort them to guarantee order
    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test)
    }

ENTITY_POOLS = {
    "hi": _build_pool(HINDI_MALE, HINDI_FEMALE),
    "ne": _build_pool(NEPALI_MALE, NEPALI_FEMALE)
}

def assign_entities(language: str, split: str, abstract_ids: List[str], seed: int) -> Dict[str, str]:
    """
    Deterministically assigns a real language-specific name to each abstract ID.
    Must ensure no duplicate names within the same example.
    """
    pool = ENTITY_POOLS[language][split]
    
    # Use deterministic random with the example's seed + language string
    # to ensure stability but different choices for hi/ne if desired (or same logic).
    rng_seed = int(hashlib.md5(f"{seed}_{language}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(rng_seed)
    
    # Sample names without replacement
    chosen_names = rng.sample(pool, k=len(abstract_ids))
    
    mapping = {}
    for eid, name in zip(abstract_ids, chosen_names):
        mapping[eid] = name
        
    return mapping
