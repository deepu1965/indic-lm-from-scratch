import json
import sys
import os

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def realize_hi(r):
    em = r["entity_mapping"]
    vals = r.get("values", {})
    attr = r["attribute"]
    repr_type = r["representation"]
    facts = r["facts"]
    question = r["question"]
    ground_truth = r["ground_truth"]
    tid = r["template_id"][-2:]
    family = r["task_family"]
    
    from phase3.scripts.dataset.audit_dataset import TEMPLATE_CONTRACT
    contract_key = (r["language"], family, r["template_id"])
    if family not in ["largest_smallest"]:
        if contract_key not in TEMPLATE_CONTRACT:
            raise ValueError(f"Missing template contract for {contract_key}")
        contract_dir = TEMPLATE_CONTRACT[contract_key]
    else:
        contract_dir = "greater"
    
    
    attr_hi = {"height": "ऊँचाई", "age": "उम्र", "price": "कीमत", "quantity": "मात्रा"}[attr]
    unit_hi = {"height": "सेमी", "age": "वर्ष", "price": "रुपये", "quantity": "इकाई"}[attr]
    
    fact_strs = []
    if repr_type == "numerical":
        for eid, name in em.items():
            v = vals.get(eid, "?")
            fact_strs.append(f"{name} की {attr_hi} {v} {unit_hi} है")
    else:
        for fact in facts:
            subj = em[fact["subject"]]
            obj = em[fact["object"]]
            rel = fact["relation"]
            if rel == "greater_than":
                fact_strs.append(f"{subj} की {attr_hi} {obj} से अधिक है")
            elif rel == "less_than":
                fact_strs.append(f"{subj} की {attr_hi} {obj} से कम है")
            elif rel == "equal":
                fact_strs.append(f"{subj} और {obj} की {attr_hi} बराबर है")

    combined_facts = "। ".join(fact_strs) + "।"
    
    if family == "direct_comparison":
        if tid == "01": combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "02":
            if len(fact_strs) >= 2: 
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"{fact_strs[0]} है, जबकि {fact_strs[1]}{rest}"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "03":
            if len(fact_strs) >= 2:
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"दी गई जानकारी के अनुसार, {fact_strs[0]}। इसके अतिरिक्त, {fact_strs[1]}{rest}"
            else: combined_facts = f"दी गई जानकारी के अनुसार, {fact_strs[0]}।"
        elif tid == "04":
            if repr_type == "numerical" and len(em) >= 2:
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"यहाँ, {fact_strs[0]}, और {fact_strs[1]}{rest}"
            else: combined_facts = "। ".join(fact_strs) + "।"
                
    elif family == "largest_smallest":
        if tid == "01": combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "02": combined_facts = f"हमें निम्नलिखित जानकारी दी गई है: {', '.join(fact_strs)}।"
        elif tid == "03":
            if len(fact_strs) >= 3: combined_facts = "। ".join(fact_strs[:-1]) + f", तथा {fact_strs[-1]}।"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "04": combined_facts = f"ज्ञात तथ्यों के आधार पर: {', '.join(fact_strs)}।"
            
    elif family == "general_comparison":
        if repr_type == "relational" and len(facts) == 1:
            fact = facts[0]
            subj = em[fact["subject"]]
            obj = em[fact["object"]]
            rel = fact["relation"]
            if tid == "01": pass
            elif tid == "02":
                if rel == "greater_than": combined_facts = f"{obj} की तुलना में {subj} की {attr_hi} अधिक है।"
                elif rel == "less_than": combined_facts = f"{obj} की तुलना में {subj} की {attr_hi} कम है।"
                elif rel == "equal": combined_facts = f"{subj} और {obj} की {attr_hi} बराबर है।"
            elif tid == "03":
                if rel == "greater_than": combined_facts = f"{attr_hi} के मामले में, {subj}, {obj} से अधिक है।"
                elif rel == "less_than": combined_facts = f"{attr_hi} के मामले में, {subj}, {obj} से कम है।"
                elif rel == "equal": combined_facts = f"{attr_hi} के मामले में, {subj} और {obj} की {attr_hi} बराबर है।"
            elif tid == "04":
                if rel == "greater_than": combined_facts = f"यह दिया गया है कि {subj} की {attr_hi} {obj} से अधिक है।"
                elif rel == "less_than": combined_facts = f"यह दिया गया है कि {subj} की {attr_hi} {obj} से कम है।"
                elif rel == "equal": combined_facts = f"यह दिया गया है कि {subj} और {obj} की {attr_hi} बराबर है।"

    elif family in ["two_hop_transitivity", "three_hop_transitivity"]:
        if tid == "01": combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "02":
            if len(fact_strs) >= 2: combined_facts = ", और ".join(fact_strs[:-1]) + f", तथा {fact_strs[-1]}।"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "03": combined_facts = f"यह ज्ञात है कि {fact_strs[0]}। साथ ही, " + "। ".join(fact_strs[1:]) + "।"
        elif tid == "04": combined_facts = f"दी गई जानकारी यह है कि {fact_strs[0]}, " + ", और ".join(fact_strs[1:]) + "।"
            
    elif family == "reversed_relation":
        if tid == "01": combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "02":
            if len(fact_strs) >= 2:
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"{fact_strs[0]}, और साथ ही {fact_strs[1]}{rest}"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "03":
            if len(fact_strs) >= 2:
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"{fact_strs[0]}। वहीं दूसरी ओर, {fact_strs[1]}{rest}"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "04":
            if len(fact_strs) >= 2:
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"इस स्थिति में, {fact_strs[0]}, तथा {fact_strs[1]}{rest}"
            else: combined_facts = "। ".join(fact_strs) + "।"

    elif family == "mixed_equality":
        if tid == "01": combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "02":
            if repr_type == "relational" and len(facts) >= 2:
                subj = em[facts[0]["subject"]]
                obj = em[facts[0]["object"]]
                if facts[0]["relation"] == "equal":
                    combined_facts = f"{subj} और {obj} दोनों की {attr_hi} बराबर है। " + "। ".join(fact_strs[1:]) + "।"
                else: combined_facts = "। ".join(fact_strs) + "।"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "03":
            if repr_type == "relational" and len(facts) >= 2:
                subj = em[facts[0]["subject"]]
                obj = em[facts[0]["object"]]
                if facts[0]["relation"] == "equal":
                    combined_facts = f"{subj} की {attr_hi} {obj} के समान है। " + "। ".join(fact_strs[1:]) + "।"
                else: combined_facts = "। ".join(fact_strs) + "।"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "04":
            if repr_type == "relational" and len(facts) >= 2:
                subj = em[facts[0]["subject"]]
                obj = em[facts[0]["object"]]
                if facts[0]["relation"] == "equal":
                    combined_facts = f"यह दिया गया है कि {subj} और {obj} की {attr_hi} बराबर है। " + "। ".join(fact_strs[1:]) + "।"
                else: combined_facts = "। ".join(fact_strs) + "।"
            else: combined_facts = "। ".join(fact_strs) + "।"

    q_entities = [em[e] for e in question["entities"]]
    q_type = question["type"]
    src_target = r["target_text"]
    rel = ground_truth.get("relation", "")
    
    if q_type == "compare":
        if tid == "01":
            if rel == "equal": q_text = f"क्या {q_entities[0]} और {q_entities[1]} की {attr_hi} बराबर है?"
            elif contract_dir == "less": q_text = f"{q_entities[0]} और {q_entities[1]} में से किसकी {attr_hi} कम है?"
            else: q_text = f"{q_entities[0]} और {q_entities[1]} में से किसकी {attr_hi} अधिक है?"
        elif tid == "02":
            if rel == "equal": q_text = f"बताइए, क्या {q_entities[0]} और {q_entities[1]} की {attr_hi} समान है?"
            elif contract_dir == "less": q_text = f"बताइए, {q_entities[0]} और {q_entities[1]} में किसकी {attr_hi} कम है?"
            else: q_text = f"बताइए, {q_entities[0]} और {q_entities[1]} में किसकी {attr_hi} अधिक है?"
        elif tid == "03":
            if rel == "equal": q_text = f"क्या {q_entities[0]} और {q_entities[1]} की {attr_hi} एक समान है?"
            elif contract_dir == "less": q_text = f"इनमें से किसकी {attr_hi} कम है: {q_entities[0]} या {q_entities[1]}?"
            else: q_text = f"इनमें से किसकी {attr_hi} अधिक है: {q_entities[0]} या {q_entities[1]}?"
        elif tid == "04":
            if rel == "equal": q_text = f"{q_entities[0]} और {q_entities[1]} की तुलना करने पर, क्या उनकी {attr_hi} बराबर है?"
            elif contract_dir == "less": q_text = f"{q_entities[0]} और {q_entities[1]} की तुलना करने पर किसकी {attr_hi} कम है?"
            else: q_text = f"{q_entities[0]} और {q_entities[1]} की तुलना करने पर किसकी {attr_hi} अधिक है?"
    elif q_type == "max":
        if tid == "01": q_text = f"इनमें से किसकी {attr_hi} सबसे अधिक है?"
        elif tid == "02": q_text = f"बताइए कि सबसे अधिक {attr_hi} किसकी है?"
        elif tid == "03": q_text = f"इन सभी में से सबसे अधिक {attr_hi} किसकी है?"
        elif tid == "04": q_text = f"सबसे अधिक {attr_hi} किसकी है?"
    elif q_type == "min":
        if tid == "01": q_text = f"इनमें से किसकी {attr_hi} सबसे कम है?"
        elif tid == "02": q_text = f"बताइए कि सबसे कम {attr_hi} किसकी है?"
        elif tid == "03": q_text = f"इन सभी में से सबसे कम {attr_hi} किसकी है?"
        elif tid == "04": q_text = f"सबसे कम {attr_hi} किसकी है?"
    else:
        q_text = f"{q_entities[0]} और {q_entities[1]} में से किसकी {attr_hi} अधिक है?"
    
    input_text = combined_facts + " " + q_text
    return input_text, src_target


def realize_ne(r):
    em = r["entity_mapping"]
    vals = r.get("values", {})
    attr = r["attribute"]
    repr_type = r["representation"]
    facts = r["facts"]
    question = r["question"]
    ground_truth = r["ground_truth"]
    tid = r["template_id"][-2:]
    family = r["task_family"]
    
    from phase3.scripts.dataset.audit_dataset import TEMPLATE_CONTRACT
    contract_key = (r["language"], family, r["template_id"])
    if family not in ["largest_smallest"]:
        if contract_key not in TEMPLATE_CONTRACT:
            raise ValueError(f"Missing template contract for {contract_key}")
        contract_dir = TEMPLATE_CONTRACT[contract_key]
    else:
        contract_dir = "greater"
    
    
    attr_ne = {"height": "उचाइ", "age": "उमेर", "price": "मूल्य", "quantity": "मात्रा"}[attr]
    unit_ne = {"height": "सेमी", "age": "वर्ष", "price": "रुपैयाँ", "quantity": "एकाइ"}[attr]
    
    fact_strs = []
    if repr_type == "numerical":
        for eid, name in em.items():
            v = vals.get(eid, "?")
            fact_strs.append(f"{name}को {attr_ne} {v} {unit_ne} छ")
    else:
        for fact in facts:
            subj = em[fact["subject"]]
            obj = em[fact["object"]]
            rel = fact["relation"]
            if rel == "greater_than":
                fact_strs.append(f"{subj}को {attr_ne} {obj} भन्दा बढी छ")
            elif rel == "less_than":
                fact_strs.append(f"{subj}को {attr_ne} {obj} भन्दा कम छ")
            elif rel == "equal":
                fact_strs.append(f"{subj} र {obj}को {attr_ne} बराबर छ")

    combined_facts = "। ".join(fact_strs) + "।"
    
    if family == "direct_comparison":
        if tid == "01": combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "02":
            if len(fact_strs) >= 2: 
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"{fact_strs[0]} छ, जबकि {fact_strs[1]}{rest}"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "03":
            if len(fact_strs) >= 2:
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"दिइएको जानकारी अनुसार, {fact_strs[0]}। यसका साथै, {fact_strs[1]}{rest}"
            else: combined_facts = f"दिइएको जानकारी अनुसार, {fact_strs[0]}।"
        elif tid == "04":
            if repr_type == "numerical" and len(em) >= 2:
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"यहाँ, {fact_strs[0]}, र {fact_strs[1]}{rest}"
            else: combined_facts = "। ".join(fact_strs) + "।"
                
    elif family == "largest_smallest":
        if tid == "01": combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "02": combined_facts = f"हामीसँग निम्न जानकारी छ: {', '.join(fact_strs)}।"
        elif tid == "03":
            if len(fact_strs) >= 3: combined_facts = "। ".join(fact_strs[:-1]) + f", तथा {fact_strs[-1]}।"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "04": combined_facts = f"ज्ञात तथ्यहरूको आधारमा: {', '.join(fact_strs)}।"
            
    elif family == "general_comparison":
        if repr_type == "relational" and len(facts) == 1:
            fact = facts[0]
            subj = em[fact["subject"]]
            obj = em[fact["object"]]
            rel = fact["relation"]
            if tid == "01": pass
            elif tid == "02":
                if rel == "greater_than": combined_facts = f"{obj}को तुलनामा {subj}को {attr_ne} बढी छ।"
                elif rel == "less_than": combined_facts = f"{obj}को तुलनामा {subj}को {attr_ne} कम छ।"
                elif rel == "equal": combined_facts = f"{subj} र {obj}को {attr_ne} बराबर छ।"
            elif tid == "03":
                if rel == "greater_than": combined_facts = f"{attr_ne}को मामलामा, {subj}, {obj} भन्दा बढी छ।"
                elif rel == "less_than": combined_facts = f"{attr_ne}को मामलामा, {subj}, {obj} भन्दा कम छ।"
                elif rel == "equal": combined_facts = f"{attr_ne}को मामलामा, {subj} र {obj}को {attr_ne} बराबर छ।"
            elif tid == "04":
                if rel == "greater_than": combined_facts = f"यो दिइएको छ कि {subj}को {attr_ne} {obj} भन्दा बढी छ।"
                elif rel == "less_than": combined_facts = f"यो दिइएको छ कि {subj}को {attr_ne} {obj} भन्दा कम छ।"
                elif rel == "equal": combined_facts = f"यो दिइएको छ कि {subj} र {obj}को {attr_ne} बराबर छ।"

    elif family in ["two_hop_transitivity", "three_hop_transitivity"]:
        if tid == "01": combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "02":
            if len(fact_strs) >= 2: combined_facts = ", र ".join(fact_strs[:-1]) + f", तथा {fact_strs[-1]}।"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "03": combined_facts = f"यो ज्ञात छ कि {fact_strs[0]}। साथै, " + "। ".join(fact_strs[1:]) + "।"
        elif tid == "04": combined_facts = f"दिइएको जानकारी यो हो कि {fact_strs[0]}, " + ", र ".join(fact_strs[1:]) + "।"
            
    elif family == "reversed_relation":
        if tid == "01": combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "02":
            if len(fact_strs) >= 2:
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"{fact_strs[0]}, र साथै {fact_strs[1]}{rest}"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "03":
            if len(fact_strs) >= 2:
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"{fact_strs[0]}। अर्कोतर्फ, {fact_strs[1]}{rest}"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "04":
            if len(fact_strs) >= 2:
                rest = ("। " + "। ".join(fact_strs[2:]) + "।") if len(fact_strs) > 2 else "।"
                combined_facts = f"यस अवस्थामा, {fact_strs[0]}, तथा {fact_strs[1]}{rest}"
            else: combined_facts = "। ".join(fact_strs) + "।"

    elif family == "mixed_equality":
        if tid == "01": combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "02":
            if repr_type == "relational" and len(facts) >= 2:
                subj = em[facts[0]["subject"]]
                obj = em[facts[0]["object"]]
                if facts[0]["relation"] == "equal":
                    combined_facts = f"{subj} र {obj} दुवैको {attr_ne} बराबर छ। " + "। ".join(fact_strs[1:]) + "।"
                else: combined_facts = "। ".join(fact_strs) + "।"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "03":
            if repr_type == "relational" and len(facts) >= 2:
                subj = em[facts[0]["subject"]]
                obj = em[facts[0]["object"]]
                if facts[0]["relation"] == "equal":
                    combined_facts = f"{subj}को {attr_ne} {obj}को समान छ। " + "। ".join(fact_strs[1:]) + "।"
                else: combined_facts = "। ".join(fact_strs) + "।"
            else: combined_facts = "। ".join(fact_strs) + "।"
        elif tid == "04":
            if repr_type == "relational" and len(facts) >= 2:
                subj = em[facts[0]["subject"]]
                obj = em[facts[0]["object"]]
                if facts[0]["relation"] == "equal":
                    combined_facts = f"यो दिइएको छ कि {subj} र {obj}को {attr_ne} बराबर छ। " + "। ".join(fact_strs[1:]) + "।"
                else: combined_facts = "। ".join(fact_strs) + "।"
            else: combined_facts = "। ".join(fact_strs) + "।"

    q_entities = [em[e] for e in question["entities"]]
    q_type = question["type"]
    src_target = r["target_text"]
    rel = ground_truth.get("relation", "")
    
    if q_type == "compare":
        if tid == "01":
            if rel == "equal": q_text = f"के {q_entities[0]} र {q_entities[1]} को {attr_ne} बराबर छ?"
            elif contract_dir == "less": q_text = f"{q_entities[0]} र {q_entities[1]} मध्ये कसको {attr_ne} कम छ?"
            else: q_text = f"{q_entities[0]} र {q_entities[1]} मध्ये कसको {attr_ne} बढी छ?"
        elif tid == "02":
            if rel == "equal": q_text = f"बताउनुहोस्, के {q_entities[0]} र {q_entities[1]} को {attr_ne} समान छ?"
            elif contract_dir == "less": q_text = f"बताउनुहोस्, {q_entities[0]} र {q_entities[1]} मध्ये कसको {attr_ne} कम छ?"
            else: q_text = f"बताउनुहोस्, {q_entities[0]} र {q_entities[1]} मध्ये कसको {attr_ne} बढी छ?"
        elif tid == "03":
            if rel == "equal": q_text = f"के {q_entities[0]} र {q_entities[1]} को {attr_ne} समान छ?"
            elif contract_dir == "less": q_text = f"यी मध्ये कसको {attr_ne} कम छ: {q_entities[0]} वा {q_entities[1]}?"
            else: q_text = f"यी मध्ये कसको {attr_ne} बढी छ: {q_entities[0]} वा {q_entities[1]}?"
        elif tid == "04":
            if rel == "equal": q_text = f"{q_entities[0]} र {q_entities[1]} तुलना गर्दा, के उनीहरूको {attr_ne} बराबर छ?"
            elif contract_dir == "less": q_text = f"{q_entities[0]} र {q_entities[1]} तुलना गर्दा कसको {attr_ne} कम छ?"
            else: q_text = f"{q_entities[0]} र {q_entities[1]} तुलना गर्दा कसको {attr_ne} बढी छ?"
    elif q_type == "max":
        if tid == "01": q_text = f"यी मध्ये कसको {attr_ne} सबैभन्दा बढी छ?"
        elif tid == "02": q_text = f"बताउनुहोस् कि सबैभन्दा बढी {attr_ne} कसको छ?"
        elif tid == "03": q_text = f"यी सबै मध्ये सबैभन्दा बढी {attr_ne} कसको छ?"
        elif tid == "04": q_text = f"सबैभन्दा बढी {attr_ne} कसको छ?"
    elif q_type == "min":
        if tid == "01": q_text = f"यी मध्ये कसको {attr_ne} सबैभन्दा कम छ?"
        elif tid == "02": q_text = f"बताउनुहोस् कि सबैभन्दा कम {attr_ne} कसको छ?"
        elif tid == "03": q_text = f"यी सबै मध्ये सबैभन्दा कम {attr_ne} कसको छ?"
        elif tid == "04": q_text = f"सबैभन्दा कम {attr_ne} कसको छ?"
    else:
        q_text = f"{q_entities[0]} र {q_entities[1]} मध्ये कसको {attr_ne} बढी छ?"
        
    input_text = combined_facts + " " + q_text
    return input_text, src_target

if __name__ == "__main__":
    lang = sys.argv[1]
    batch_id = sys.argv[2]
    
    with open(f"scratch/full_generation/{lang}/prompts/{batch_id}.json", "r", encoding="utf-8") as f:
        recs = json.load(f)
    
    failed_file = f"scratch/full_generation/{lang}/failed/{batch_id}.txt"
    if os.path.exists(failed_file):
        failed_ids = set(open(failed_file).read().split())
        recs = [r for r in recs if r["id"] in failed_ids]
    
    realize_fn = realize_hi if lang == "hi" else realize_ne
    
    results = []
    for r in recs:
        input_text, target_text = realize_fn(r)
        results.append({"id": r["id"], "input_text": input_text, "target_text": target_text})
    
    os.makedirs(f"scratch/full_generation/{lang}/responses", exist_ok=True)
    out_file = f"scratch/full_generation/{lang}/responses/{batch_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(results)} records -> {out_file}")
