import json
import time
import asyncio
import argparse
import numpy as np
from openai import AsyncOpenAI

async def process_prompt(client, model, prompt_data, gen_params, concurrency_sem):
    async with concurrency_sem:
        start_time = time.time()
        first_token_time = None
        output_tokens = 0
        generated_text = ""

        try:
            # Pedido à API do vLLM
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_data["text"]}],
                max_tokens=128, 
                temperature=gen_params.get("temperature", 0.0),
                top_p=gen_params.get("top_p", 1.0),
                seed=gen_params.get("seed", 42),
                stream=True
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    generated_text += chunk.choices[0].delta.content
                    output_tokens += 1

            end_time = time.time()
            ttft = first_token_time - start_time if first_token_time else 0
            tpot = (end_time - first_token_time) / max(1, output_tokens - 1) if first_token_time and output_tokens > 1 else 0
            total_time = end_time - start_time

            return {
                "prompt_id": prompt_data.get("id", "unknown"),
                "category": prompt_data.get("category", "unknown"),
                "mandatory": prompt_data.get("mandatory", False),
                "status": 200,
                "error": None,
                "ttft_s": ttft,
                "tpot_s": tpot,
                "total_s": total_time,
                "output_token_events": output_tokens,
                "throughput_output_tok_s": output_tokens / total_time if total_time > 0 else 0,
                "generated_text": generated_text
            }
        except Exception as e:
            return {
                "prompt_id": prompt_data.get("id", "unknown"),
                "category": prompt_data.get("category", "unknown"),
                "status": 500,
                "error": str(e)
            }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()

    # Ler o JSON com a estrutura do Track A
    with open(args.prompts, "r", encoding="utf-8") as f:
        data = json.load(f)

    # CORREÇÃO CHAVE: Validar se a chave 'prompts' existe e extrair a lista!
    if "prompts" in data:
        prompts_list = data["prompts"]
    elif isinstance(data, list):
        prompts_list = data
    else:
        print("[ERRO] Formato do JSON não reconhecido. Falta a chave 'prompts'.")
        return

    gen_params = data.get("generation", {})

    client = AsyncOpenAI(api_key="sk-nada", base_url=args.url)
    sem = asyncio.Semaphore(args.concurrency)

    print(f"[Cliente] A iniciar {len(prompts_list)} pedidos (Concorrência: {args.concurrency})...")

    # Executar as chamadas
    tasks = [process_prompt(client, args.model, p, gen_params, sem) for p in prompts_list]
    results = await asyncio.gather(*tasks)

    # Escrever ficheiro bruto JSONL (Igual ao do teu amigo)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Gerar Summary por Categoria
    sucesso = [r for r in results if r["status"] == 200]
    categorias = set(r["category"] for r in sucesso)
    summary = []

    for cat in sorted(categorias):
        cat_results = [r for r in sucesso if r["category"] == cat]
        summary.append({
            "category": cat,
            "runs": len(cat_results),
            "successful_runs": len(cat_results),
            "ttft_s_mean": np.mean([r["ttft_s"] for r in cat_results]) if cat_results else 0,
            "ttft_s_stdev": np.std([r["ttft_s"] for r in cat_results]) if len(cat_results) > 1 else 0.0,
            "tpot_s_mean": np.mean([r["tpot_s"] for r in cat_results]) if cat_results else 0,
            "tpot_s_stdev": np.std([r["tpot_s"] for r in cat_results]) if len(cat_results) > 1 else 0.0,
            "throughput_output_tok_s_mean": np.mean([r["throughput_output_tok_s"] for r in cat_results]) if cat_results else 0,
            "throughput_output_tok_s_stdev": np.std([r["throughput_output_tok_s"] for r in cat_results]) if len(cat_results) > 1 else 0.0
        })

    # Escrever o ficheiro Summary formatado
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Cliente] Concluído com Sucesso!")

if __name__ == "__main__":
    asyncio.run(main())