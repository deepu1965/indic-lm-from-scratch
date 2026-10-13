from typing import Dict, List, Any, Optional

class Template:
    def __init__(self, template_id: str, language: str, task_family: str,
                 fact_formats: Dict[str, str],
                 question_formats: Dict[str, str],
                 answer_formats: Dict[str, str],
                 question_direction: str = "greater"):
        self.template_id = template_id
        self.language = language
        self.task_family = task_family
        self.fact_formats = fact_formats
        self.question_formats = question_formats
        self.answer_formats = answer_formats
        self.question_direction = question_direction
        
    def render_fact(self, subject: str, object_name: str, relation: str, attr_lex: Dict) -> str:
        if relation == "greater_than":
            rel_word = attr_lex["fact_greater"]
        elif relation == "less_than":
            rel_word = attr_lex["fact_less"]
        else:
            rel_word = attr_lex["fact_equal"]
            
        fmt = self.fact_formats.get("relational", "{subject} की {attr} {object_name} {rel_word} है।")
        return fmt.format(subject=subject, object_name=object_name, attr=attr_lex["noun"], rel_word=rel_word)
        
    def render_num_fact(self, subject: str, val: Any, attr_lex: Dict) -> str:
        fmt = self.fact_formats.get("numerical", "{subject} की {attr} {val} {unit} है।")
        return fmt.format(subject=subject, attr=attr_lex["noun"], val=val, unit=attr_lex.get("unit", ""))

    def render_question(self, q_type: str, entities: List[str], attr_lex: Dict) -> str:
        if q_type == "compare":
            fmt = self.question_formats["compare"]
            # Some templates ask for greater, some for less, some general
            adj = attr_lex["adj_greater"] if self.question_direction == "greater" else attr_lex["adj_less"]
            if self.question_direction == "general":
                adj = "" # Not used in general formats
            return fmt.format(e1=entities[0], e2=entities[1], attr=attr_lex["noun"], adj=adj)
        elif q_type == "largest":
            fmt = self.question_formats["largest"]
            return fmt.format(entities=", ".join(entities), attr=attr_lex["noun"], adj=attr_lex["adj_greater"])
        elif q_type == "smallest":
            fmt = self.question_formats["smallest"]
            return fmt.format(entities=", ".join(entities), attr=attr_lex["noun"], adj=attr_lex["adj_less"])
        return ""

# We will dynamically build the 28 HI and 28 NE templates in a tailored way.
TEMPLATES_HI = []
TEMPLATES_NE = []

def _build_hi_templates():
    # Helper to generate task-specific templates
    families = [
        "direct_comparison", "general_comparison", "largest_smallest", 
        "two_hop_transitivity", "three_hop_transitivity", "reversed_relation", "mixed_equality"
    ]
    
    for tf in families:
        prefix = "".join([w[0].upper() for w in tf.split("_")])
        for i in range(4):
            tid = f"HI_{prefix}_0{i+1}"
            
            # Default formats
            f_fmt = {"relational": "{subject} की {attr} {object_name} {rel_word} है।", "numerical": "{subject} की {attr} {val} {unit} है।"}
            q_fmt = {"compare": "{e1} और {e2} में से किसकी {attr} {adj} है?", "largest": "इनमें से किसकी {attr} सबसे {adj} है?", "smallest": "इनमें से किसकी {attr} सबसे {adj} है?"}
            a_fmt = {"greater": "{ans} की {attr} अधिक है।", "less": "{ans} की {attr} कम है。", "equal": "{e1} और {e2} की {attr} बराबर है।", "largest": "{ans} की {attr} सबसे अधिक है।", "smallest": "{ans} की {attr} सबसे कम है।"}
            
            q_dir = "greater" if i % 2 == 0 else "less"
            
            if tf == "direct_comparison":
                if i == 1:
                    f_fmt["relational"] = "हमें ज्ञात है कि {subject} की {attr} {object_name} {rel_word} है।"
                    q_dir = "greater"
                elif i == 2:
                    q_fmt["compare"] = "बताइए कि {e1} और {e2} में किसकी {attr} {adj} है।"
                    q_dir = "less"
                elif i == 3:
                    q_fmt["compare"] = "प्रश्न यह है कि {e1} और {e2} में से किसकी {attr} {adj} है?"
                    q_dir = "greater"
                    
            elif tf == "general_comparison":
                q_dir = "general"
                q_fmt["compare"] = ["{e1} और {e2} की {attr} की तुलना करें।", "{e1} की तुलना में {e2} की {attr} कैसी है?", "विस्तार से बताएँ कि {e1} और {e2} में क्या अंतर है।", "{e1} और {e2} की स्थिति की तुलना करें।"][i]
                # For general comparison, target defaults to greater if not equal
                
            elif tf == "largest_smallest":
                q_dir = "greater" # handeled by q_type dynamically anyway
                if i == 1:
                    q_fmt["largest"] = "दिए गए लोगों में से किसकी {attr} सबसे {adj} है?"
                    q_fmt["smallest"] = "दिए गए लोगों में से किसकी {attr} सबसे {adj} है?"
                elif i == 2:
                    q_fmt["largest"] = "पता लगाएँ कि किसकी {attr} सबसे {adj} है।"
                    q_fmt["smallest"] = "पता लगाएँ कि किसकी {attr} सबसे {adj} है।"
                elif i == 3:
                    f_fmt["numerical"] = "जानकारी के अनुसार {subject} की {attr} {val} {unit} है।"
                    
            elif tf == "two_hop_transitivity":
                if i == 0:
                    q_fmt["compare"] = "श्रृंखला के आधार पर, {e1} और {e2} में से किसकी {attr} {adj} है?"
                elif i == 1:
                    q_dir = "greater"
                    q_fmt["compare"] = "अंततः {e1} और {e2} में से किसकी {attr} {adj} है?"
                elif i == 2:
                    q_dir = "less"
                elif i == 3:
                    q_fmt["compare"] = "निष्कर्ष निकालें कि {e1} और {e2} में किसकी {attr} {adj} है।"
                    
            elif tf == "three_hop_transitivity":
                if i == 0:
                    q_fmt["compare"] = "सभी तथ्यों को मिलाकर, {e1} और {e2} में किसकी {attr} {adj} है?"
                elif i == 1:
                    q_dir = "greater"
                elif i == 2:
                    q_dir = "less"
                    q_fmt["compare"] = "बताएँ कि {e1} और {e2} में किसकी {attr} {adj} है।"
                elif i == 3:
                    q_dir = "general"
                    q_fmt["compare"] = "{e1} और {e2} की {attr} की तुलना करें।"
                    
            elif tf == "reversed_relation":
                if i == 0:
                    q_fmt["compare"] = "तो {e1} और {e2} में से किसकी {attr} {adj} है?"
                elif i == 1:
                    q_dir = "greater"
                elif i == 2:
                    q_dir = "less"
                elif i == 3:
                    q_dir = "general"
                    q_fmt["compare"] = "{e1} और {e2} की {attr} की तुलना करें।"
                    
            elif tf == "mixed_equality":
                if i == 0:
                    q_fmt["compare"] = "समानता को देखते हुए, {e1} और {e2} में किसकी {attr} {adj} है?"
                elif i == 1:
                    q_dir = "greater"
                elif i == 2:
                    q_dir = "less"
                elif i == 3:
                    q_dir = "general"
                    q_fmt["compare"] = "समानता के आधार पर, {e1} और {e2} की {attr} की तुलना करें।"
            
            t = Template(tid, "hi", tf, f_fmt, q_fmt, a_fmt, q_dir)
            TEMPLATES_HI.append(t)

def _build_ne_templates():
    families = [
        "direct_comparison", "general_comparison", "largest_smallest", 
        "two_hop_transitivity", "three_hop_transitivity", "reversed_relation", "mixed_equality"
    ]
    
    for tf in families:
        prefix = "".join([w[0].upper() for w in tf.split("_")])
        for i in range(4):
            tid = f"NE_{prefix}_0{i+1}"
            
            f_fmt = {"relational": "{subject}को {attr} {object_name}को {rel_word} छ।", "numerical": "{subject}को {attr} {val} {unit} छ।"}
            q_fmt = {"compare": "{e1} र {e2}मध्ये कसको {attr} {adj} छ?", "largest": "यीमध्ये कसको {attr} सबैभन्दा {adj} छ?", "smallest": "यीमध्ये कसको {attr} सबैभन्दा {adj} छ?"}
            a_fmt = {"greater": "{ans}को {attr} बढी छ।", "less": "{ans}को {attr} कम छ।", "equal": "{e1} र {e2}को {attr} बराबर छ।", "largest": "{ans}को {attr} सबैभन्दा बढी छ।", "smallest": "{ans}को {attr} सबैभन्दा कम छ।"}
            
            q_dir = "greater" if i % 2 == 0 else "less"
            
            if tf == "direct_comparison":
                if i == 1:
                    f_fmt["relational"] = "हामीलाई थाहा छ कि {subject}को {attr} {object_name}को {rel_word} छ।"
                    q_dir = "greater"
                elif i == 2:
                    q_fmt["compare"] = "बताउनुहोस् कि {e1} र {e2}मध्ये कुनको {attr} {adj} छ।"
                    q_dir = "less"
                elif i == 3:
                    q_fmt["compare"] = "प्रश्न यो छ कि {e1} र {e2}मध्ये कसको {attr} {adj} छ?"
                    q_dir = "greater"
                    
            elif tf == "general_comparison":
                q_dir = "general"
                q_fmt["compare"] = ["{e1} र {e2}को {attr} तुलना गर्नुहोस्।", "{e1}को तुलनामा {e2}को {attr} कस्तो छ?", "{e1} र {e2}मध्ये के फरक छ बताउनुहोस्।", "{e1} र {e2}को स्थितिको तुलना गर्नुहोस्।"][i]
                
            elif tf == "largest_smallest":
                q_dir = "greater"
                if i == 1:
                    q_fmt["largest"] = "दिइएका व्यक्तिहरूमध्ये कसको {attr} सबैभन्दा {adj} छ?"
                    q_fmt["smallest"] = "दिइएका व्यक्तिहरूमध्ये कसको {attr} सबैभन्दा {adj} छ?"
                elif i == 2:
                    q_fmt["largest"] = "पत्ता लगाउनुहोस् कि कसको {attr} सबैभन्दा {adj} छ।"
                    q_fmt["smallest"] = "पत्ता लगाउनुहोस् कि कसको {attr} सबैभन्दा {adj} छ।"
                elif i == 3:
                    f_fmt["numerical"] = "जानकारी अनुसार {subject}को {attr} {val} {unit} छ।"
                    
            elif tf == "two_hop_transitivity":
                if i == 0:
                    q_fmt["compare"] = "श्रृङ्खलाको आधारमा, {e1} र {e2}मध्ये कसको {attr} {adj} छ?"
                elif i == 1:
                    q_dir = "greater"
                    q_fmt["compare"] = "अन्त्यमा {e1} र {e2}मध्ये कुनको {attr} {adj} छ?"
                elif i == 2:
                    q_dir = "less"
                elif i == 3:
                    q_fmt["compare"] = "निष्कर्ष निकाल्नुहोस् कि {e1} र {e2}मध्ये कुनको {attr} {adj} छ।"
                    
            elif tf == "three_hop_transitivity":
                if i == 0:
                    q_fmt["compare"] = "सबै तथ्यहरू मिलाएर, {e1} र {e2}मध्ये कसको {attr} {adj} छ?"
                elif i == 1:
                    q_dir = "greater"
                elif i == 2:
                    q_dir = "less"
                    q_fmt["compare"] = "बताउनुहोस् कि {e1} र {e2}मध्ये कसको {attr} {adj} छ।"
                elif i == 3:
                    q_dir = "general"
                    q_fmt["compare"] = "{e1} र {e2}को {attr} तुलना गर्नुहोस्।"
                    
            elif tf == "reversed_relation":
                if i == 0:
                    q_fmt["compare"] = "त्यसोभए {e1} र {e2}मध्ये कसको {attr} {adj} छ?"
                elif i == 1:
                    q_dir = "greater"
                elif i == 2:
                    q_dir = "less"
                elif i == 3:
                    q_dir = "general"
                    q_fmt["compare"] = "{e1} र {e2}को {attr} तुलना गर्नुहोस्।"
                    
            elif tf == "mixed_equality":
                if i == 0:
                    q_fmt["compare"] = "समानतालाई हेर्दा, {e1} र {e2}मध्ये कसको {attr} {adj} छ?"
                elif i == 1:
                    q_dir = "greater"
                elif i == 2:
                    q_dir = "less"
                elif i == 3:
                    q_dir = "general"
                    q_fmt["compare"] = "समानताको आधारमा, {e1} र {e2}को {attr} तुलना गर्नुहोस्।"
            
            t = Template(tid, "ne", tf, f_fmt, q_fmt, a_fmt, q_dir)
            TEMPLATES_NE.append(t)

_build_hi_templates()
_build_ne_templates()

def get_templates(language: str) -> List[Template]:
    return TEMPLATES_HI if language == "hi" else TEMPLATES_NE
