from mlc_llm import MLCEngine

# Inicializa o motor com o nosso modelo local e força o uso do CPU
print("A carregar o modelo...")
engine = MLCEngine(
    model="/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion/TinyLlama-MLC", 
    device="cpu"
)

# Faz uma pergunta ao modelo
prompt = "Explain what a supercomputer is in two short sentences."
print(f"\nPergunta: {prompt}\n")
print("A gerar resposta...\n")

# Gera a resposta (igual ao ChatGPT)
response = engine.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100
)

# Imprime o resultado final
print("Resposta:")
print(response.choices[0].message.content)
print("\n>>> TESTE CONCLUÍDO COM SUCESSO!")