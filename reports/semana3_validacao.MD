# Semana 3 — Validação da Estratégia

## Visão Geral

Durante a Semana 3, o objetivo foi avaliar se a estratégia baseada em cruzamento de médias móveis simples (SMA) apresenta poder preditivo real ou se seu desempenho pode ser explicado por aleatoriedade ou sobreajuste (overfitting).

Para isso, foram aplicadas diversas técnicas de validação amplamente utilizadas em pesquisas quantitativas:

- análise de estabilidade de parâmetros
- validação walk-forward
- estimativa da Probabilidade de Overfitting de Backtest (PBO)
- comparação com estratégias aleatórias
- comparação com um benchmark passivo (Buy & Hold)

A estratégia analisada ao longo dos experimentos foi um cruzamento de médias móveis aplicado ao ativo BTCUSDT em dados de 1 hora.

Configuração base da estratégia:

SMA(20,100)

Capital inicial: \$10.000

Custos de transação e slippage foram incluídos em todas as simulações.

---

# 1. Estabilidade de Parâmetros

Para avaliar a sensibilidade da estratégia aos parâmetros, foram testadas variações próximas da configuração base:

- SMA(18,100)
- SMA(20,100)
- SMA(22,100)
- SMA(20,90)
- SMA(20,110)

O objetivo foi verificar se a estratégia apresenta bom desempenho apenas em uma combinação muito específica de parâmetros — o que indicaria possível sobreajuste — ou se existe uma região mais ampla de desempenho aceitável.

### Observação

A análise revelou a existência de uma **região de desempenho razoavelmente estável**, em vez de um único ponto ótimo.

Isso sugere que a estratégia **não depende exclusivamente de uma combinação exata de parâmetros**, o que é um indício positivo de robustez.

Entretanto, as diferenças de desempenho entre as configurações foram relativamente moderadas, indicando que o possível edge da estratégia pode ser limitado.

---

# 2. Validação Walk-Forward

Foi implementado um processo de validação walk-forward com janelas móveis, simulando um cenário mais próximo do uso real de uma estratégia quantitativa.

O procedimento foi o seguinte:

1. otimizar parâmetros em um período histórico (janela de treino)
2. aplicar a configuração escolhida em dados futuros não utilizados na otimização (janela de teste)
3. repetir o processo ao longo do tempo com janelas móveis

### Resumo dos Resultados

| Janela | Configuração Vencedora | Sharpe (Treino) | Sharpe (Teste) |
|------|------|------|------|
| 1 | SMA(18/100) | 1.86 | -0.52 |
| 2 | SMA(18/100) | 1.29 | -0.45 |
| 3 | SMA(24/80) | 1.02 | 0.30 |
| 4 | SMA(24/80) | 1.14 | -1.62 |
| 5 | SMA(24/80) | -0.43 | 0.03 |

### Interpretação

A estratégia **não manteve desempenho consistentemente positivo fora da amostra**.

Embora algumas janelas tenham apresentado resultados razoáveis, outras exibiram Sharpe negativo.

Isso indica que a estratégia **não generaliza bem entre diferentes regimes de mercado**.

---

# 3. Probabilidade de Overfitting de Backtest (PBO)

Para estimar a probabilidade de que a melhor configuração tenha sido selecionada apenas por sorte estatística, foi conduzida uma análise inspirada no conceito de PBO.

Metodologia:

1. dividir os dados históricos em vários blocos
2. selecionar parâmetros ótimos em alguns blocos
3. testar esses parâmetros em blocos fora da amostra
4. observar com que frequência a configuração vencedora falha fora da amostra

### Resultado

Probabilidade estimada de overfitting:

66,7%

### Interpretação

Um valor de PBO acima de 50% sugere que a escolha dos parâmetros pode ser fortemente influenciada por seleção estatística, em vez de refletir um verdadeiro sinal preditivo.

Esse resultado indica **alto risco de sobreajuste no processo de seleção de parâmetros**.

---

# 4. Benchmark com Estratégias Aleatórias

Para verificar se a estratégia possui algum poder preditivo real, ela foi comparada com um conjunto de estratégias aleatórias.

Uma melhoria importante foi calibrar as estratégias aleatórias para possuir **nível de turnover semelhante ao da estratégia SMA**, garantindo uma comparação mais justa.

Simulação realizada:

- 300 estratégias aleatórias
- mesmos custos de transação
- frequência de negociação equivalente

### Resultados

| Métrica | SMA | Média Random |
|------|------|------|
| Sharpe | 0.449 | 0.021 |
| CAGR | 9.8% | -3.9% |

A estratégia SMA superou **aproximadamente 81% das estratégias aleatórias**.

### Interpretação

Esse resultado sugere que a estratégia **captura alguma estrutura real no mercado** e não se comporta como uma regra puramente aleatória.

---

# 5. Comparação com Buy & Hold

A estratégia também foi comparada com um benchmark passivo.

| Estratégia | Sharpe | CAGR | Drawdown Máximo |
|------|------|------|------|
| SMA | 0.449 | 9.8% | -45.8% |
| Buy & Hold | 0.591 | 18.3% | -50.1% |

### Interpretação

A estratégia Buy & Hold superou a SMA tanto em Sharpe quanto em CAGR.

Por outro lado, a SMA apresentou um drawdown ligeiramente menor.

Isso sugere que a estratégia baseada em médias móveis atua mais como **um filtro de risco parcial do que como uma estratégia de geração de retornos superiores ao mercado**.

---

# 6. Avaliação Final

Os resultados do processo de validação permitem algumas conclusões importantes.

### A estratégia possui edge?

Há evidências de **um edge fraco**.

A estratégia supera a maioria das estratégias aleatórias e apresenta alguma estabilidade de parâmetros.

No entanto, esse edge parece limitado.

---

### A estratégia é robusta?

Parcialmente.

Embora exista alguma estabilidade de parâmetros, a análise walk-forward revelou **grande variabilidade de desempenho ao longo do tempo**.

Isso indica dependência significativa do regime de mercado.

---

### A estratégia sobrevive fora da amostra?

Não de forma consistente.

Várias janelas walk-forward apresentaram Sharpe negativo.

Isso sugere que o poder preditivo da estratégia **não se mantém estável ao longo do tempo**.

---

### O resultado foi apenas sorte?

A comparação com estratégias aleatórias indica que a estratégia não é equivalente a um processo puramente aleatório.

No entanto, o alto valor de PBO e a instabilidade fora da amostra sugerem que **seleção estatística provavelmente desempenhou papel relevante nos resultados observados**.

---

# Conclusão Final

A estratégia de cruzamento de médias móveis apresenta **um sinal fraco, porém detectável**, superando a maior parte das estratégias aleatórias quando avaliadas sob condições realistas de negociação.

Contudo, a estratégia não supera o benchmark passivo Buy & Hold e demonstra robustez limitada quando testada fora da amostra.

Portanto, embora a estratégia capture alguma estrutura do mercado, **ela não apresenta atualmente um edge suficientemente forte para ser utilizada como estratégia autônoma confiável**.

Trabalhos futuros podem incluir:

- detecção de regimes de mercado
- inclusão de variáveis adicionais
- construção de modelos multifatoriais
- avaliação em períodos históricos mais longos