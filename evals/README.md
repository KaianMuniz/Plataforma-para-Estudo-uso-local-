## Instruções

## Monte um conjunto rotulado

1. Escolha uma coleção representativa e escreva perguntas em português que cubram seus assuntos e formulações comuns.
2. Inclua perguntas respondíveis, parcialmente respondíveis e sem resposta nos arquivos, inclusive perguntas claramente fora do assunto.
3. Para cada pergunta, atribua `expected_status`: `sufficient`, `partial` ou `insufficient`. Faça a classificação manualmente, olhando os arquivos, antes de ver a resposta do modelo.
4. Para perguntas respondíveis ou parciais, rotule ao menos um trecho que sustente parte da resposta usando `source_id` e `chunk_number` dos registros Chroma. Use `grade: 2` para evidência direta e `grade: 1` para contexto útil, mas secundário. Perguntas sem resposta devem ter lista vazia.
5. Mantenha perguntas de calibração e de validação separadas. Não ajuste o prompt ou escolha o reranker olhando só para as perguntas de validação.

Comece com um conjunto equilibrado (por exemplo, 10–20 perguntas por classe) e aumente-o conforme surgirem erros. Esse número é uma sugestão operacional, não um limiar estatístico de aprovação. Salve suas perguntas reais em `evals/local_questions.jsonl`; o `.gitignore` exclui esse padrão para não publicar perguntas ou dados privados acidentalmente. `questions.example.jsonl` mostra o formato e contém IDs fictícios que precisam ser substituídos.

## Execute a avaliação

Com o ambiente virtual do app ativo, Ollama aberto, modelos já baixados e a coleção preenchida:

```powershell
python evals/evaluate_answers.py --collection "Biologia" --questions evals/local_questions.jsonl --output evals/baseline.json
```

Para medir o reranker opcional, primeiro instale suas dependências e rode o mesmo conjunto:

```powershell
pip install -r requirements-reranker.txt
python evals/evaluate_answers.py --collection "Biologia" --questions evals/local_questions.jsonl --reranker-model BAAI/bge-reranker-v2-m3 --output evals/reranked.json
```

O modelo BGE reranker v2 M3 é multilingual e compara diretamente o par pergunta–trecho. Ele reordena os candidatos retornados pelo RRF; não substitui Chroma, BM25 ou RRF e não decide sozinho se existe evidência suficiente. 

## Resultados

- `accuracy` compara o estado previsto pelo fluxo com o rótulo humano.
- `false_accept_rate_on_unanswerable` é a proporção de perguntas rotuladas sem resposta que acabaram como `partial` ou `sufficient`. Examine cada falso aceite; esta métrica é especialmente importante para avaliar a recusa.
- `mean_recall_at_5` mede quantos trechos relevantes rotulados aparecem entre os cinco recuperados.
- `mean_ndcg_at_5` considera também a posição e o grau de relevância de cada trecho.
- `confusion_matrix` mostra quais classes são confundidas. O JSON de saída contém perguntas, respostas e trechos recuperados para inspeção manual.

Compare `baseline.json` e `reranked.json` usando as mesmas perguntas e os mesmos rótulos. Além das métricas, leia os falsos aceites, perguntas parciais classificadas como suficientes e respostas cuja citação não sustente a afirmação. Adote o reranker somente se a melhora se repetir em perguntas de validação e compensar o custo de memória e tempo no computador alvo. Não selecione um corte de score com base em um único exemplo ou conjunto.


