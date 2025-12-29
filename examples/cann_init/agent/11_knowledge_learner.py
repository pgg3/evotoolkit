# evotoolkit-master\examples\cann_init\agent\knowledge_learner.py
#!/usr/bin/env python3
# Copyright (c) 2025 Yansong Sun
# Licensed under the MIT License
import json
from evotoolkit.evo_method.cann_initer.parsers import parse_json
import os
from pathlib import Path
from _config import (
    get_llm, get_knowledge_base, get_test_config, load_python_ref,
    ensure_output_dir, KNOWLEDGE_CANDIDATES_DIR
)

class KnowledgeLearner:
    def __init__(self, kb_path="knowledge_base.json", candidates_dir=KNOWLEDGE_CANDIDATES_DIR):
        self.kb_path = Path(kb_path)
        self.candidates_dir = Path(candidates_dir)
        self.llm = get_llm()

    def load_current_kb(self):
        if self.kb_path.exists():
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_kb(self, rules):
        with open(self.kb_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)

    def learn(self):
        """主学习流程"""
        if not self.candidates_dir.exists():
            print(f"[Learner] Candidate dir not found: {self.candidates_dir}")
            return

        current_rules = self.load_current_kb()
        # 建立标题索引，方便 Merge 更新
        rule_map = {r['title']: r for r in current_rules}

        files = list(self.candidates_dir.glob("*.json"))
        # 排除已处理的
        files = [f for f in files if not f.name.endswith(".processed")]

        if not files:
            print("[Learner] No new candidates found.")
            return

        print(f"[Learner] Found {len(files)} new candidates. Starting analysis...")

        for sample_file in files:
            print(f"\n[Learner] Processing {sample_file.name}...")
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample = json.load(f)

                # 核心: 调用 LLM 进行模式匹配与决策
                decision = self._analyze_pattern(sample, current_rules)

                if decision:
                    action = decision.get("action")
                    if action == "NEW":
                        new_rule = decision.get("content")
                        if new_rule and new_rule.get("title") not in rule_map:
                            print(f"  [Action: NEW] + {new_rule.get('title')}")
                            current_rules.append(new_rule)
                            rule_map[new_rule['title']] = new_rule  # Update index
                        else:
                            print(f"  [Action: NEW] Skipped (Title collision)")

                    elif action == "MERGE":
                        target_title = decision.get("target_title")
                        updated_content = decision.get("content")
                        if target_title in rule_map:
                            print(f"  [Action: MERGE] Updating '{target_title}'...")
                            # 覆盖旧规则的内容 (Feature/Fix/Reason 可能会被 LLM 优化)
                            rule_map[target_title].update(updated_content)
                            # 保持 title 不变，防止 LLM 瞎改
                            rule_map[target_title]['title'] = target_title
                        else:
                            print(f"  [Action: MERGE] Target '{target_title}' not found, treating as SKIP.")

                    elif action == "SKIP":
                        print(f"  [Action: SKIP] {decision.get('reason', 'No reason')}")

                # 标记已处理
                sample_file.rename(sample_file.with_suffix('.json.processed'))

            except Exception as e:
                print(f"  [Learner Error] Failed to process {sample_file}: {e}")
                import traceback
                traceback.print_exc()

        # 全部处理完后保存
        # Reconstruct list from map to ensure updates are saved
        final_rules = list(rule_map.values())
        self.save_kb(final_rules)
        print(f"\n[Learner] Knowledge Base updated. Total rules: {len(final_rules)}")

    def _analyze_pattern(self, sample, current_rules):
        """
        利用 LLM 将具体 Case 转化为通用 Pattern，并决定是 Merge 还是 New
        """
        raw_error_dict = sample.get('raw_error', {})
        # 截取错误日志防止 Token 爆炸
        raw_error_str = str(raw_error_dict.get('error', ''))[:1500]
        diagnosis = sample.get('effective_diagnosis', '')
        strategy = sample.get('effective_strategy', '')
        # [Fixed] 这里的 Key 必须与 dump_success_sample 一致
        prev_stage = sample.get('prev_stage', 'unknown')
        curr_stage = sample.get('curr_stage', 'unknown')

        # [Fixed] 获取更精确的分类信息
        expert_category = sample.get('error_category', sample.get('suggestion_type', 'unknown'))
        expert_action = sample.get('action_decision', sample.get('meta', {}).get('fix_agent', 'unknown'))

        # 将现有知识库压缩为 Context
        kb_context = ""
        for idx, r in enumerate(current_rules):
            kb_context += f"Rule {idx}: [{r['title']}]\n   Feature: {r['feature']}\n"

        prompt = f"""
You are the Chief Knowledge Architect for an Ascend C Operator Development System.
Your goal is to maintain a high-quality, non-redundant, pattern-driven Knowledge Base (KB).
## [重要] Environment: 
CANN 8.1.rc1, Ascend 910B. Use strictly modern 'Ascend C' APIs (namespace AscendC). Avoid deprecated TBE or Tik syntax.

## [Task]
Analyze the provided "Success Sample" (a resolved error case) and decide how to update the "Current KB".

## [Input: Success Sample]
- **Raw Error**: {raw_error_str}
- **Expert Diagnosis**: {diagnosis}
- **Expert Classification**: {expert_category} 
- **Action Taken**: {expert_action}           
- **Effective Fix**: {strategy}
- **Stage Transformation**: the 'error stage' transform from **{prev_stage}** to **{curr_stage}**

## [Input: Current KB Index]
{kb_context}

## [Decision Logic - VITAL]
Compare the Sample against the Current KB.
1. **MATCH/MERGE**: If the error is a variation of an existing Rule (even if the error message is slightly different, but the *root cause* is the same pattern), you MUST choose to **MERGE**.
   - *Example*: Error "get_block_num undefined" is just a symptom of "Host-Device Boundary Violation". Merge it!
   - *Action*: Improve the existing rule's 'feature' list to include this new symptom, or refine the 'fix' to be more robust.
2. **NEW**: If the error represents a *fundamentally new* coding pattern or architectural constraint not covered by any existing rule.
   - *Action*: Create a new Rule.
3. **SKIP**: If the error is trivial, random noise, or too specific to be generalized.

## [Output Requirements]
- **Language**: Use **Chinese** for the content of the rule (Title, Feature, Reason, Fix).
- **Style**: Developer-centric, Pattern-driven. Focus on "Why architecture forbids this" not "What the compiler said".
- **Format**: JSON ONLY.

## [JSON Schema]
{{
    "action": "MERGE" | "NEW" | "SKIP",
    "target_title": "Existing Rule Title (Only if MERGE)",
    "reason": "Why you chose this action (for log)",
    "content": {{
        "title": "Standardized Title (e.g. Host-Device Boundary Violation)",
        "feature": "Updated list of error signatures (Generalized, e.g. 'Contains macros like GET_TILING or host functions')",
        "reason": "Deep architectural explanation (e.g. 'Kernel runs on AI Core, Host runs on CPU...')",
        "fix": "Universal coding guideline (e.g. 'Decouple Host/Device code, use raw pointers')",
        "type": "Must be consistent with the Expert Classification (e.g. {expert_category}) or one of: REMOVE_HEADER, USE_RAW_POINTER, FIX_LOGIC..."
    }}
}}
"""
        response, _ = self.llm.get_response(prompt)
        try:
            return parse_json(response)
        except:
            print("[Learner] Failed to parse LLM response")
            return None

if __name__ == "__main__":
    learner = KnowledgeLearner()
    learner.learn()