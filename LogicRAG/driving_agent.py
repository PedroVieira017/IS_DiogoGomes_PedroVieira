import os
import json
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def obter_ultimo_facto_csv(caminho_ficheiro):
    """
    Abre o ficheiro CSV do LogicRAG e extrai a última linha registada.
    """
    try:
        with open(caminho_ficheiro, mode='r', encoding='utf-8') as ficheiro:
            leitor = csv.reader(ficheiro)
            linhas = list(leitor)

            if len(linhas) > 1:
                # pega na última linha do ficheiro (ignora o cabeçalho)
                ultima_linha = linhas[-1]
                return ", ".join(ultima_linha)
            else:
                return "Nenhum facto detetado no ficheiro."

    except FileNotFoundError:
        print(f"Aviso: O ficheiro '{caminho_ficheiro}' não foi encontrado.")
        print("A usar o facto de teste por defeito...\n")
        return "Pedestrian detected at crosswalk. Distance: 100 meters. Ego-vehicle speed: 30km/h."
    except Exception as e:
        return f"Erro ao ler CSV: {e}"


def guardar_resposta_csv(caminho_ficheiro, facto, resposta):
    """
    Guarda o facto de entrada e a resposta do LLM num CSV de histórico.
    Cria o ficheiro com cabeçalho se ainda não existir.
    """
    ficheiro_existe = os.path.isfile(caminho_ficheiro)

    with open(caminho_ficheiro, mode='a', newline='', encoding='utf-8') as ficheiro:
        escritor = csv.writer(ficheiro)

        # só escreve o cabeçalho na primeira vez
        if not ficheiro_existe:
            escritor.writerow(["timestamp", "facto_logicrag", "resposta_llm"])

        escritor.writerow([datetime.now().isoformat(), facto, resposta])


def criar_prompt(logic_rag_result):
    return f"""Responde em portugues de Portugal.
Tarefa academica offline: classificar uma cena gravada do dataset KITTI.
Isto nao controla um veiculo real e nao e aconselhamento operacional em tempo real.

Escolhe uma das seguintes etiquetas de decisao de alto nivel:
- STOP: parar ou preparar paragem imediata.
- SLOW_DOWN: abrandar e manter vigilancia reforcada.
- CONTINUE: continuar apenas se nao houver risco relevante.

Regras:
- Se houver peoes a aproximar-se do veiculo, escolhe uma decisao conservadora.
- Nao sugiras manobras evasivas ou desvio de trajetoria.
- Responde apenas com a etiqueta escolhida e duas frases de justificacao.

O motor logico LogicRAG detetou o seguinte facto na estrada:
{logic_rag_result}

Classifica a cena e justifica a decisao."""


def chamar_iaedu(prompt, api_key, api_endpoint, channel_id, thread_id):
    print("Sending data to IAedu API...\n")
    print(f"Using IAedu thread: {thread_id}\n")

    # the iaedu.pt endpoint expects multipart/form-data, not JSON
    multipart_data = {
        "message": (None, prompt),
        "thread_id": (None, thread_id),  # groups conversation context
        "channel_id": (None, channel_id),
        "user_info": (None, "{}")
    }

    response = requests.post(
        api_endpoint,
        headers={
            "x-api-key": api_key  # iaedu uses x-api-key, not Bearer
        },
        files=multipart_data,
        stream=True,  # response comes back as SSE chunks
        timeout=60,
    )

    response.raise_for_status()

    resposta_completa = []

    # read SSE stream line by line and print tokens as they arrive
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            decoded = decoded[len("data: "):]

        try:
            chunk = json.loads(decoded)
            event_type = chunk.get("type")

            if event_type == "token":
                token = chunk.get("content", "")
                if token:
                    print(token, end="", flush=True)
                    resposta_completa.append(token)
            elif event_type == "done":
                break  # server signals end of stream

        except json.JSONDecodeError:
            pass  # skip heartbeat lines or malformed chunks

    return "".join(resposta_completa)


def chamar_ollama(prompt):
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")

    print("Sending data to local Ollama model...\n")
    print(f"Using Ollama model: {model}\n")

    response = requests.post(
        endpoint,
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
            },
        },
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    resposta_completa = []
    for line in response.iter_lines():
        if not line:
            continue

        chunk = json.loads(line.decode("utf-8"))
        token = chunk.get("response", "")
        if token:
            print(token, end="", flush=True)
            resposta_completa.append(token)

        if chunk.get("done"):
            break

    return "".join(resposta_completa).strip()


def main():
    print("Starting LogicRAG + LLM Integration System...\n")

    # pull credentials from .env so we don't hardcode anything sensitive
    # As credenciais ficam fora da imagem e são lidas apenas do volume local.
    load_dotenv(BASE_DIR / ".env")
    llm_backend = os.getenv("LLM_BACKEND", "iaedu").strip().lower()
    api_key = os.getenv("OPENAI_API_KEY")
    api_endpoint = os.getenv("OPENAI_API_ENDPOINT")
    channel_id = os.getenv("IAEDU_CHANNEL_ID")
    thread_id = os.getenv("IAEDU_THREAD_ID") or f"logic_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if llm_backend not in {"iaedu", "ollama"}:
        print("Error: LLM_BACKEND must be 'iaedu' or 'ollama'.")
        return 1

    if llm_backend == "iaedu" and (not api_key or not api_endpoint or not channel_id):
        print("Error: Missing credentials in .env file.")
        return 1

    # read the latest detected scene fact from the LogicRAG output CSV
    logic_rag_result = obter_ultimo_facto_csv(BASE_DIR / "resultados_kitti.csv")
    print(f"Fact extracted by LogicRAG: \n'{logic_rag_result}'\n")

    prompt = criar_prompt(logic_rag_result)

    try:
        print("Assistant Response:")
        print("-" * 50)

        if llm_backend == "ollama":
            resposta_texto = chamar_ollama(prompt)
        else:
            resposta_texto = chamar_iaedu(prompt, api_key, api_endpoint, channel_id, thread_id)

        print("\n" + "-" * 50)

        # save input fact + LLM response to the history CSV
        guardar_resposta_csv(BASE_DIR / "logic_rag_response.csv", logic_rag_result, resposta_texto)
        print("Resposta guardada em logic_rag_response.csv")
        return 0

    except requests.exceptions.HTTPError as e:
        print(f"\nAPI Failure [{e.response.status_code}]: {e.response.text}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
