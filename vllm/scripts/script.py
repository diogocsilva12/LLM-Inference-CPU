import json
import time
from vllm import LLM, SamplingParams

def main():
    ficheiro_entrada = "dataset_entrada.json"
    ficheiro_saida = "resultados_vllm_opt.json"

    # 1. Carregar ou criar o ficheiro de entrada JSON
    try:
        with open(ficheiro_entrada, "r", encoding="utf-8") as f:
            perguntas = json.load(f)
    except FileNotFoundError:
        print(f"[Aviso] Ficheiro '{ficheiro_entrada}' não encontrado. A criar um dataset de teste automático...")
        perguntas = [
            "Olá, como estás?",
            "Explain what a supercomputer is.",
            "Translate 'Hello world' to French.",
            "What is the theory of relativity?",
            "Write a short Python code to reverse a string."
        ]
        with open(ficheiro_entrada, "w", encoding="utf-8") as f:
            json.dump(perguntas, f, indent=4)

    print(f"\n[>] A iniciar vLLM. Total de perguntas a processar: {len(perguntas)}")

    # 2. Inicializar o LLM para CPU
    # Removi o 'gpu_memory_utilization' porque em CPU ele é ignorado e atira avisos.
    llm = LLM(
        model='facebook/opt-125m',
        enforce_eager=True
    )

    # 3. Definir parâmetros da resposta (aumentei max_tokens para respostas melhores)
    sampling_params = SamplingParams(max_tokens=150, temperature=0.7)

    # 4. A MAGIA DO BATCHING: Passas a lista inteira de uma vez!
    print("A gerar respostas em batch usando todos os cores...")
    start_time = time.time()
    
    outputs = llm.generate(perguntas, sampling_params)
    
    end_time = time.time()

    # 5. Estruturar os resultados para gravar no JSON
    resultados_finais = []
    for i, output in enumerate(outputs):
        resultados_finais.append({
            "id": i,
            "prompt": output.prompt,
            # output.outputs[0].text contém o texto gerado final
            "resposta": output.outputs[0].text.strip()
        })

    # 6. Gravar em disco
    with open(ficheiro_saida, "w", encoding="utf-8") as f:
        json.dump(resultados_finais, f, indent=4, ensure_ascii=False)

    # 7. Imprimir estatísticas de performance
    tempo_total = end_time - start_time
    throughput = len(perguntas) / tempo_total
    
    print("\n" + "="*50)
    print(f" TESTE CONCLUÍDO COM SUCESSO!")
    print(f"️ Tempo Total: {round(tempo_total, 2)} segundos")
    print(f" Throughput:  {round(throughput, 2)} perguntas / segundo")
    print(f" Guardado em: {ficheiro_saida}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()