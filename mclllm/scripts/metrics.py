import json
import asyncio
import time
from mlc_llm import AsyncMLCEngine

async def processar_perguntas(modelo_path, batch_size, perguntas):
    print(f"\n[>] A carregar Modelo: {modelo_path} | Batch Size Máx: {batch_size}")

    # 1. Inicializa o motor no modo "server" para permitir batching agressivo
    engine = AsyncMLCEngine(
        model=modelo_path,
        device="cpu",
        mode="server",
        engine_config={"max_batch_size": batch_size}
    )

    # Função auxiliar que processa uma única pergunta
    async def gerar_resposta(idx, prompt):
        t_inicio = time.time()
        # O motor trata de colocar este pedido na "fila" do batch automaticamente
        resposta = await engine.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        t_fim = time.time()
        
        return {
            "id": idx,
            "prompt": prompt,
            "resposta": resposta.choices[0].message.content,
            "tempo_segundos": round(t_fim - t_inicio, 2)
        }

    print(f"A enviar {len(perguntas)} perguntas em simultâneo para o motor...")
    start_total = time.time()

    # 2. Lança todas as perguntas ao mesmo tempo! O C++ agrupa-as no Batch Size definido.
    tarefas = [gerar_resposta(i, p) for i, p in enumerate(perguntas)]
    resultados = await asyncio.gather(*tarefas)

    end_total = time.time()
    throughput = len(perguntas) / (end_total - start_total)
    
    print(f"[!] Processamento concluído em {round(end_total - start_total, 2)} segundos!")
    print(f"[!] Throughput: {round(throughput, 2)} perguntas por segundo.")

    return resultados

async def main():
    # --- PASSO A: Criar um JSON de teste falso (Só para poderes testar já) ---
    dummy_data = [
        "What is the capital of Portugal?",
        "Explain Quantum Computing in simple terms.",
        "Write a haiku about a supercomputer.",
        "What is 25 * 4?",
        "How far is the moon from Earth?",
        "Give me a recipe for pancakes.",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?"
    ]
    with open("dataset_entrada.json", "w", encoding="utf-8") as f:
        json.dump(dummy_data, f)
    # -------------------------------------------------------------------------

    # 1. Ler as perguntas do JSON
    with open("dataset_entrada.json", "r", encoding="utf-8") as f:
        perguntas = json.load(f)

    # 2. Definir a nossa "Matriz de Experiências"
    # Aqui podes adicionar outros caminhos de modelos (ex: Llama3, OPT) e testar qual o batch size ideal
    experiencias = [
        {"modelo": "./TinyLlama-MLC", "batch_size": 2},
        {"modelo": "./TinyLlama-MLC", "batch_size": 8}
    ]

    resultados_finais = {}

    # 3. Correr o motor para cada experiência
    for exp in experiencias:
        res = await processar_perguntas(exp["modelo"], exp["batch_size"], perguntas)
        
        # Guarda os resultados com um nome de chave identificativo
        chave = f"{exp['modelo']}_batch_{exp['batch_size']}"
        resultados_finais[chave] = res

    # 4. Guardar tudo num JSON estruturado para poderes analisar
    with open("resultados_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(resultados_finais, f, indent=4, ensure_ascii=False)
        
    print("\n>>> SUCESSO! Todos os resultados guardados em 'resultados_benchmark.json'")

if __name__ == "__main__":
    asyncio.run(main())