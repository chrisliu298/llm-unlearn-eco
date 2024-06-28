import time

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from eco.attack.utils import remove_hooks


class InferenceEngine:
    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        evaluator,
        batch_size=64,
        prompt_prefix="",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.subset_names = subset_names
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.prompt_prefix = prompt_prefix

    def prepare_dataset(self):
        self.datasets = {}
        for subset_name in self.subset_names:
            self.datasets[subset_name] = self.data_module.load_dataset_for_eval(
                subset_name,
                load_in_batch=True,
                batch_size=self.batch_size,
                prompt_prefix=self.prompt_prefix,
            )

    def inference(self):
        raise NotImplementedError(
            f"inference not implemented for {self.__class__.__name__}"
        )

    def summary(self):
        summary_stats, outputs = [], []
        for result in self.results:
            name, data = list(result.items())[0]
            if (
                self.data_module.dataset_type == "multiple_choice"
                and self.data_module.name != "truthfulqa"
            ):
                pred, correct = [], []
                for d in data:
                    pred.extend(d["predicted"])
                    correct.extend(d["correct"])
                data = np.array(pred) == np.array(correct)
            avg_score = {name: float(np.mean(data))}
            summary_stats.append(avg_score)
            print(avg_score)
            outputs.append(result)
        return summary_stats, outputs


class EvaluationEngine(InferenceEngine):
    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        evaluator,
        batch_size=64,
        prompt_prefix="",
    ):
        super().__init__(
            model,
            tokenizer,
            data_module,
            subset_names,
            evaluator,
            batch_size,
            prompt_prefix,
        )

    def inference(self):
        self.prepare_dataset()
        self.results = []
        for subset_name, dataset in self.datasets.items():
            all_outputs = []
            total_time, total_examples = 0, 0
            for batch in tqdm(
                dataset,
                desc=f"Evaluating {self.evaluator.name} of {self.data_module.name} on {subset_name}",
                total=len(dataset),
            ):
                remove_hooks(self.model.model)
                prompts = batch[self.data_module.eval_prompt_key]
                answers = batch[self.data_module.eval_answer_key]

                start_time = time.perf_counter()
                outputs = self.evaluator.evaluate(
                    prompts, answers, self.model, self.tokenizer
                )
                end_time = time.perf_counter()
                total_time += end_time - start_time
                total_examples += len(prompts)

                if self.data_module.dataset_type == "multiple_choice":
                    if self.data_module.name != "truthfulqa":
                        correct_answer = batch["correct_answer"]
                        outputs = [{"correct": correct_answer, "predicted": outputs}]
                all_outputs.extend(outputs)
                remove_hooks(self.model.model)
            self.results.append(
                {
                    f"{self.data_module.name}_{subset_name}_{self.evaluator.name}": all_outputs
                }
            )
            avg_time_per_example = (
                total_time / total_examples if total_examples > 0 else 0
            )
            # print(
            #     tabulate(
            #         [
            #             ["Total examples", total_examples],
            #             ["Total time (sec)", f"{total_time:.4f}"],
            #             ["Avg time (sec)", f"{avg_time_per_example:.4f}"],
            #         ],
            #         headers=[f"{subset_name} of {self.data_module.name}", "Value"],
            #         tablefmt="pretty",
            #     )
            # )
        return self.results


class GenerationEngine(InferenceEngine):
    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        evaluator,
        batch_size=64,
        prompt_prefix="",
        comparison_length=128,
        truncate_answers=False,
    ):
        super().__init__(
            model,
            tokenizer,
            data_module,
            subset_names,
            evaluator,
            batch_size,
            prompt_prefix,
        )
        self.comparison_length = comparison_length
        self.truncate_answers = truncate_answers
        if not isinstance(self.evaluator, list):
            self.evaluator = [evaluator]

    def inference(self):
        self.results = []
        answers = self._generate()
        self.text_generations = {}
        for subset_name, data in answers.items():
            # Flatten data["gold"] and data["generated"]
            data_gold = [item for sublist in data["gold"] for item in sublist]
            data_generated = [item for sublist in data["generated"] for item in sublist]
            self.text_generations[f"{self.data_module.name}_{subset_name}"] = {
                "gold": data_gold,
                "generated": data_generated,
            }
            for evaluator in self.evaluator:
                evaluator_outputs = []
                for prompt, gold, generated in tqdm(
                    zip(data["prompt"], data["gold"], data["generated"]),
                    total=len(data["gold"]),
                    desc=f"Evaluating {evaluator.name} of {self.data_module.name} on {subset_name}",
                ):
                    if evaluator.name == "perplexity":
                        generated = [p + g for p, g in zip(prompt, generated)]
                    outputs = evaluator.evaluate(gold, generated)
                    evaluator_outputs.extend(outputs)
                self.results.append(
                    {
                        f"{self.data_module.name}_{subset_name}_{evaluator.name}": evaluator_outputs
                    }
                )

    def _generate(self):
        self.prepare_dataset()
        padding_side = self.tokenizer.padding_side
        if padding_side != "left":
            self.tokenizer.padding_side = "left"
        subsets_generations = {}
        for subset_name, dataset in self.datasets.items():
            all_gold_answers, all_generated_answers = [], []
            all_prompts = []
            total_time, total_examples = 0, 0
            for batch in tqdm(
                dataset,
                desc=f"Generating completions of {self.data_module.name} on {subset_name}",
                total=len(dataset),
            ):
                remove_hooks(self.model.model)
                prompts = batch[self.data_module.gen_prompt_key]
                gold_answers = batch[self.data_module.gen_answer_key]

                tokenized_prompts = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(self.model.device)

                # Generate and decode answers
                start_time = time.perf_counter()
                generated = self.model.generate(
                    **tokenized_prompts,
                    prompts=prompts,
                    generation_config=self.model.generation_config,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                end_time = time.perf_counter()
                total_time += end_time - start_time
                total_examples += len(prompts)
                generated_answers = self.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )

                # Remove prompt from generated answers
                generated_answers_truncated = []
                for p, g in zip(prompts, generated_answers):
                    generated_answers_truncated.append(g[len(p) :])
                # Remove special tokens from answers
                gold_answers = self.tokenizer.batch_decode(
                    self.tokenizer(gold_answers, add_special_tokens=False).input_ids,
                    skip_special_tokens=True,
                )
                if self.truncate_answers:
                    gold_answers, generated_answers_truncated = self.truncate(
                        gold_answers, generated_answers_truncated
                    )
                all_gold_answers.append(gold_answers)
                all_generated_answers.append(generated_answers_truncated)
                all_prompts.append(prompts)
                remove_hooks(self.model.model)

            assert (
                len(all_gold_answers) == len(all_generated_answers) == len(all_prompts)
            ), f"Length mismatch: {len(all_gold_answers)}, {len(all_generated_answers)}, {len(all_prompts)}"
            subsets_generations[subset_name] = {
                "prompt": all_prompts,
                "gold": all_gold_answers,
                "generated": all_generated_answers,
            }

            avg_time_per_example = (
                total_time / total_examples if total_examples > 0 else 0
            )
            # print(
            #     tabulate(
            #         [
            #             ["Total examples", total_examples],
            #             ["Total time (sec)", f"{total_time:.4f}"],
            #             ["Avg time (sec)", f"{avg_time_per_example:.4f}"],
            #         ],
            #         headers=[f"{subset_name} of {self.data_module.name}", "Value"],
            #         tablefmt="pretty",
            #     )
            # )

        # Reset padding side
        self.tokenizer.padding_side = padding_side
        return subsets_generations

    def truncate(self, gold, generated):
        truncated_gold, truncated_generated = [], []
        for gold_answer, generated_answer in zip(gold, generated):
            min_len = min(len(gold_answer), len(generated_answer))
            truncated_gold.append(gold_answer[:min_len])
            truncated_generated.append(generated_answer[:min_len])
        return truncated_gold, truncated_generated
