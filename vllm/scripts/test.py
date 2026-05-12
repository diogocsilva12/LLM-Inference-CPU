from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        model='facebook/opt-125m',
        gpu_memory_utilization=0.5,
        enforce_eager=True
    )
    outputs = llm.generate(['Olá, como estás?'], SamplingParams(max_tokens=50))
    print(outputs[0].outputs[0].text)

if __name__ == '__main__':
    main()
