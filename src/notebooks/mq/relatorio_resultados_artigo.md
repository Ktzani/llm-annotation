# Relatório de Resultados — Anotação Automática de Datasets com Ensemble de LLMs Zero-Shot

> **Finalidade:** base de resultados para o artigo final da disciplina de Métodos
> Quantitativos. Consolida o que foi obtido nos notebooks em `src/notebooks/mq/`.
> **Autor:** Gabriel Catizani Faria Oliveira — DCC/UFMG.
> **Status:** todos os eixos concluídos — fatorial 2³, comparação de sistemas,
> regressão e **projeto de 1 fator** (com AGNews = 2 e DBLP = 4 réplicas).

> 📷 **INSTRUÇÃO AO CLAUDE WEB (montagem do artigo):** este relatório contém
> marcadores **`📷 [FIGURA: nome_do_arquivo.png — descrição]`** indicando onde
> entram as imagens. **Não invente nem gere essas figuras.** Ao escrever o artigo,
> **peça ao usuário cada figura, UMA POR VEZ**, na ordem em que aparecem no texto
> (cite o nome do arquivo), e só então prossiga para a próxima. Todas estão em
> `src/notebooks/mq/figuras/`. Figuras marcadas como *(opcional)* podem ser
> dispensadas se o usuário não as enviar.

> **Checklist das figuras (ordem de solicitação):**
> 1. `_prem_sem.png` — premissas sem transformação (§2.1)
> 2. `_prem_com.png` — premissas com logit (§2.1)
> 3. `P2_variacao_fatorial.png` — % variação F1 original (§2.2)
> 4. `variacao_f1logit.png` — % variação F1 logit (§2.2)
> 5. `variacao_acc.png` — % variação Acurácia original *(opcional)* (§2.2)
> 6. `variacao_acclogit.png` — % variação Acurácia logit *(opcional)* (§2.2)
> 7. `P1_forest_plot_diferenca.png` — forest plot da diferença por dataset (§3.1)
> 8. `_prem_reg_linearidade.png` — premissa de linearidade da regressão (LOESS) (§4.1)
> 9. `P3_acerto_vs_confianca.png` — taxa de acerto × confiança, por dataset (§4.1)
> 10. `_prem_reg_residuos.png` — premissas da regressão (resíduos M6) (§4.2)
> 11. `P3b_dispersao_confianca.png` — dispersão da confiança *(opcional)* (§4.2)
> 12. `_prem_um_fator.png` — premissas do projeto de 1 fator (§5.1)

---

## 1. Contexto e objetivos

Investiga-se o uso de um **ensemble de 3 LLMs compactas zero-shot** —
**Llama3.1-8B, Qwen3-8B, DeepSeek-R1-8B** — combinadas por **voto majoritário**
para anotar automaticamente datasets de classificação de texto, comparando-as a
modelos *fine-tuned* sobre o benchmark de referência. Datasets (com nº de classes):

| Dataset | Domínio | Nº de classes |
|---|---|---|
| MovieReview | Geral | 2 (binário) |
| AGNews | Geral | 4 |
| Books | Técnico | 8 |
| DBLP | Técnico | 10 |

**Perguntas de pesquisa.** **P1:** o desempenho *downstream* com rótulos do
ensemble equivale ao treinado com rótulos reais? **P2:** domínio, grau de consenso
e tamanho do texto afetam a qualidade das anotações? **P3:** a confiança do modelo
prediz a qualidade da anotação?

**Métricas.** **F1-macro** (resposta principal, controla desbalanceamento) e
**Acurácia**. Métrica de qualidade calculada sobre o consenso vs. *ground truth*.

**Definição de consenso (partição por concordância).** Para cada exemplo, entre as
3 predições: **3×0** = os 3 modelos concordam (unânime); **2×1** = exatamente 2
concordam (maioria simples). Os dois grupos **particionam** os exemplos com maioria
(`n(2×1) + n(3×0) = total`). Exemplos sem maioria (3 predições distintas) e rótulos
inválidos são descartados.

### 1.1 Mudanças em relação à proposta original — justificativas

Durante a execução, várias decisões da proposta inicial foram revistas. Em todos os
casos a mudança foi motivada por **rigor estatístico** ou **pelos próprios dados**.
O quadro abaixo resume; os detalhes estão nos eixos correspondentes.

| # | Proposta original | O que foi feito | Por quê |
|---|---|---|---|
| 1 | Métricas: Acurácia, F1-macro **e Fleiss' Kappa** | **F1-macro (principal) + Acurácia** | A concordância entre modelos já é modelada **diretamente** pelo fator/variável **Consenso (2×1 vs 3×0)** — mais interpretável e acionável (filtragem) que o Kappa agregado. O foco recai nas métricas que respondem à qualidade vs. *ground truth*. |
| 2 | Fatorial 2³ com **Fator A = Arquitetura** (sem raciocínio Llama × com raciocínio Qwen/DeepSeek) | Fator A trocado por **D = Tamanho do texto** (curto × longo) | A "arquitetura" **não é propriedade do rótulo de consenso** (a resposta): o voto majoritário combina os 3 modelos, não tem arquitetura única. Além disso o contraste proposto era **desbalanceado** (1 modelo × 2 modelos). O **tamanho do texto** é um fator binário limpo, bem definido por exemplo e plausivelmente ligado à qualidade. |
| 3 | Fatorial com **os 4 datasets como "replicações naturais"**, resposta **agregada por dataset** | **Réplicas reais** = execuções repetidas (5 *timestamps*) do mesmo dataset; resposta **por célula** | Usar datasets como réplicas **confunde réplica com o próprio fator Domínio** (os datasets *são* os níveis de B) e deixa o desenho **sem estimativa de erro experimental**. Execuções repetidas dão réplicas genuínas e erro experimental. Só **MovieReview e Books** têm 5 réplicas → são os níveis de Domínio no 2³. |
| 4 | P1 com **4 datasets como unidades pareadas** + H₀ (teste de hipótese) | Pareamento **por fold** dentro de cada dataset + **IC 95%** + **Bonferroni/BH** | Pooling de 4 datasets de dificuldade muito distinta num único teste tem **baixo poder** e esconde diferenças. Pareando **por fold** (10; 5 no AGNews) há amostras pareadas reais e **conclusão por dataset**. Como passam a ser 4 testes (um por dataset), aplicou-se **correção de múltiplas comparações**. |
| 5 | P3: **regressão linear simples**, X = log-prob médio, Y = **F1 agregado por dataset** | **LPM no nível da anotação** (~208 mil obs), resposta binária `correct`, modelos M1–M6 (EP HC1) | Regressão agregada por dataset teria **n = 4 pontos** — sem validade estatística. No nível da anotação cada exemplo tem confiança e acerto próprios → poder real; covariáveis (consenso, domínio) isolam a contribuição marginal da confiança. |
| 6 | **Calibração dos log-probs** como etapa prévia; log-prob como **critério de filtragem** | Calibração **adiada (trabalho futuro)**; **confiança abandonada como filtro** | Os dados mostraram confiança **saturada (~83% ≥ 0,99)**, relação **não-linear** e R²_aj ≤ 0,06 sozinha → nenhum limiar a separaria (ver §4). Adotou-se o **consenso** como critério de qualidade — decisão sustentada pelos dados (fator dominante no fatorial). |
| 7 | *(não previsto)* | **Verificação de premissas + transformação logit** (fatorial) e diagnósticos (regressão) | Exigência de rigor (Jain): ANOVA/regressão pressupõem normalidade, homoscedasticidade e linearidade. F1/Acurácia **violavam** normalidade/homoscedasticidade → **logit** corrige ambas e dá inferência válida. |
| 8 | *(não previsto)* | **Projeto de 1 fator (Domínio), ANOVA desbalanceada** com os 4 datasets | O Domínio no fatorial usa só 2 datasets e **confunde domínio com nº de classes**. Um projeto de 1 fator com **4 níveis** (réplicas desiguais: AGNews 2, DBLP 4) isola e detalha a **ordenação** dos domínios. |

**Observação importante (a registrar no artigo).** A pergunta **P2** da proposta
citava "arquitetura do modelo"; como o fator A foi substituído por **tamanho do
texto**, a redação de P2 deve passar a falar em **domínio, grau de consenso e
tamanho do texto** (e não arquitetura). As perguntas **P1** e **P3** permanecem,
mas **P3** muda de conclusão esperada: a confiança **não** é um bom preditor/filtro
(resultado negativo, devidamente justificado).

---

## 2. Eixo 1 — Projeto Fatorial 2³ (P2)

**Desenho.** Fatorial completo 2³ com **r = 5 réplicas** (2³ × 5 = **40
observações**). Fatores:

| Fator | −1 | +1 |
|---|---|---|
| **B — Domínio** | MovieReview (geral) | Books (técnico) |
| **C — Consenso** | Maioria simples (2×1) | Unanimidade (3×0) |
| **D — Tamanho do texto** | Curto | Longo (corte na mediana por domínio) |

**Matriz de contrastes (sinais −1/+1)** com os valores `y` das 5 réplicas,
Média(y) e o cálculo dos coeficientes: a última linha é **q = Total/8** (q₀ = média
geral). **Os valores `y` estão na escala logit** — a mesma usada em toda a análise
(as premissas só são atendidas após a transformação, §2.1). Por isso os `q` aqui
coincidem com os efeitos da §2.2.

**F1-macro (logit):**

| I | B | C | D | BC | BD | CD | BCD | y — 5 réplicas (r1…r5) | Média(y) |
|---|---|---|---|---|---|---|---|---|---|
| +1 | −1 | −1 | −1 | +1 | +1 | +1 | −1 | (0,550; 0,567; 0,579; 0,672; 0,580) | 0,590 |
| +1 | +1 | −1 | −1 | −1 | −1 | +1 | +1 | (−0,038; −0,099; 0,002; −0,088; −0,087) | −0,062 |
| +1 | −1 | +1 | −1 | −1 | +1 | −1 | +1 | (2,896; 2,885; 2,885; 2,838; 2,850) | 2,871 |
| +1 | −1 | −1 | +1 | +1 | −1 | −1 | +1 | (0,511; 0,431; 0,411; 0,565; 0,377) | 0,459 |
| +1 | +1 | +1 | −1 | +1 | −1 | −1 | −1 | (1,196; 1,220; 1,207; 1,229; 1,222) | 1,215 |
| +1 | +1 | −1 | +1 | −1 | +1 | −1 | −1 | (0,207; 0,216; 0,143; 0,143; 0,233) | 0,188 |
| +1 | −1 | +1 | +1 | −1 | −1 | +1 | −1 | (2,807; 2,769; 2,802; 2,755; 2,810) | 2,789 |
| +1 | +1 | +1 | +1 | +1 | +1 | +1 | +1 | (1,491; 1,494; 1,513; 1,513; 1,489) | 1,500 |
| **Total** | −3,867 | +7,199 | +0,323 | −2,022 | +0,748 | +0,084 | −0,013 | (Σ sinal·Média) | +9,549 |
| **q = Total/8** | **−0,483** | **+0,900** | +0,040 | −0,253 | +0,094 | +0,010 | −0,002 | q₀ = | **1,194** |

**Acurácia (logit):**

| I | B | C | D | BC | BD | CD | BCD | y — 5 réplicas (r1…r5) | Média(y) |
|---|---|---|---|---|---|---|---|---|---|
| +1 | −1 | −1 | −1 | +1 | +1 | +1 | −1 | (0,554; 0,570; 0,588; 0,672; 0,587) | 0,594 |
| +1 | +1 | −1 | −1 | −1 | −1 | +1 | +1 | (0,069; 0,008; 0,098; 0,026; 0,038) | 0,048 |
| +1 | −1 | +1 | −1 | −1 | +1 | −1 | +1 | (2,897; 2,886; 2,886; 2,839; 2,851) | 2,872 |
| +1 | −1 | −1 | +1 | +1 | −1 | −1 | +1 | (0,513; 0,438; 0,413; 0,573; 0,378) | 0,463 |
| +1 | +1 | +1 | −1 | +1 | −1 | −1 | −1 | (1,320; 1,350; 1,335; 1,356; 1,357) | 1,344 |
| +1 | +1 | −1 | +1 | −1 | +1 | −1 | −1 | (0,105; 0,117; 0,051; 0,050; 0,106) | 0,086 |
| +1 | −1 | +1 | +1 | −1 | −1 | +1 | −1 | (2,807; 2,769; 2,802; 2,755; 2,810) | 2,789 |
| +1 | +1 | +1 | +1 | +1 | +1 | +1 | +1 | (1,332; 1,333; 1,352; 1,341; 1,333) | 1,338 |
| **Total** | −3,902 | +7,151 | −0,182 | −2,055 | +0,246 | +0,005 | −0,091 | (Σ sinal·Média) | +9,533 |
| **q = Total/8** | **−0,488** | **+0,894** | −0,023 | −0,257 | +0,031 | +0,001 | −0,011 | q₀ = | **1,192** |

*(Linhas dos tratamentos na ordem (1), b, c, d, bc, bd, cd, bcd. Valores `y` em
**logit**. **O efeito é o coeficiente q** (convenção de Jain); q₀ na coluna I = média
geral (em logit). A inferência — IC e contrastes — é feita sobre **q**. % de variação
e significância na §2.2.)*

**Erros experimentais por réplica (cálculo do SSE).** O valor previsto de cada
tratamento é a **média das suas réplicas** ($\hat{y}$); o erro de cada réplica é
$e_{ij}=y_{ij}-\hat{y}$. Por construção, **os erros somam 0 em cada tratamento**, e
$SSE=\sum e_{ij}^2$ (escala logit, F1):

| Trat. | ŷ (média) | erros e_ij das 5 réplicas | Σ e_ij | Σ e_ij² |
|---|---|---|---|---|
| (1) | 0,590 | (−0,039; −0,023; −0,011; +0,083; −0,010) | ≈0 | 0,0091 |
| b | −0,062 | (+0,024; −0,037; +0,064; −0,026; −0,025) | ≈0 | 0,0074 |
| c | 2,871 | (+0,025; +0,014; +0,014; −0,033; −0,021) | ≈0 | 0,0025 |
| d | 0,459 | (+0,052; −0,028; −0,048; +0,106; −0,082) | ≈0 | 0,0239 |
| bc | 1,215 | (−0,019; +0,005; −0,008; +0,014; +0,007) | ≈0 | 0,0007 |
| bd | 0,188 | (+0,019; +0,028; −0,045; −0,046; +0,045) | ≈0 | 0,0073 |
| cd | 2,789 | (+0,019; −0,019; +0,013; −0,034; +0,022) | ≈0 | 0,0025 |
| bcd | 1,500 | (−0,009; −0,006; +0,013; +0,013; −0,011) | ≈0 | 0,0006 |
| **SSE** | | | | **0,0541** |

Logo $s_e^2 = SSE/[2^k(r-1)] = 0{,}0541/32 = 1{,}69\times10^{-3}$ (g.l. = 32) — é a
base do IC dos efeitos (§2.2) e dos contrastes (§2.4). *(Acurácia logit: mesmo
procedimento, SSE = 0,0490 → $s_e^2 = 1{,}53\times10^{-3}$.)* O erro é minúsculo
(réplicas quase idênticas), o que torna os ICs estreitos.

> Apenas **MovieReview e Books** possuem 5 réplicas; por isso o fatorial usa esses
> dois domínios. As respostas por célula são F1-macro e Acurácia do consenso.

### 2.1 Verificação de premissas (antes da análise)

Resíduos $e_{ij} = y_{ij}-\bar{y}_{\text{tratamento}}$. Testes (Shapiro-Wilk,
Levene robusto), n = 40:

| Resposta | Normalidade (Shapiro p) | Homoscedasticidade (Levene p) | Veredito |
|---|---|---|---|
| F1 (original) | **0,021** | 0,059 | normalidade **violada** |
| Acurácia (original) | **0,007** | **0,024** | ambas **violadas** |
| **logit(F1)** | **0,218** | **0,287** | **OK** |
| **logit(Acurácia)** | **0,111** | **0,160** | **OK** |

**Diagnóstico visual:** QQ-plot em "S" (caudas pesadas) e **funil** em
resíduos×ajustados (cells 2×1 variam ~5× mais que 3×0). A **transformação logit
corrige as duas violações** — logo, a inferência estatisticamente válida é a do
espaço **logit**.

> 📷 **[FIGURA: `_prem_sem.png` — diagnósticos das premissas SEM transformação
> (histograma, QQ-plot e resíduos×ajustados de F1 e Acurácia): mostra o "S" e o funil]**
>
> 📷 **[FIGURA: `_prem_com.png` — mesmos diagnósticos COM logit: caudas e funil
> corrigidos]**

### 2.2 Efeitos (escala logit — inferência por IC 95%)

O **efeito** é o coeficiente **q** (convenção de Jain); significância pelo **IC 95%**
(significativo ⇔ o IC **não contém 0**); % da variação explicada (SS/SST). O IC de
cada efeito segue o slide: $q_i \pm t_{[0{,}975;\,32]}\cdot s_{q_i}$, com
$s_{q_i}=\sqrt{s_e^2/N}$ e g.l. do erro $=2^k(r-1)=32$.

**F1-macro (logit):**

| Efeito | q (efeito) | IC 95% de q | Signif. | % variação |
|---|---|---|---|---|
| **C — Consenso** | **+0,900** | [+0,887; +0,913] | ✓ | **72,4%** |
| **B — Domínio** | **−0,483** | [−0,497; −0,470] | ✓ | **20,9%** |
| **BC** | −0,253 | [−0,266; −0,240] | ✓ | 5,7% |
| BD | +0,094 | [+0,080; +0,107] | ✓ | 0,8% |
| D — Tamanho | +0,040 | [+0,027; +0,054] | ✓ | 0,2% |
| CD | +0,010 | [−0,003; +0,024] | ✗ | 0,0% |
| BCD | −0,002 | [−0,015; +0,012] | ✗ | 0,0% |

**Acurácia (logit):** mesmo padrão — **C q=+0,894 [+0,881; +0,907] (72,3%)**,
**B q=−0,488 [−0,500; −0,475] (21,5%)**, **BC q=−0,257 [−0,269; −0,244] (6,0%)**;
D, BD pequenos e significativos; **CD e BCD não-significativos** (IC contém 0).
Erro experimental ≈ **0,1–0,3%** (réplicas quase determinísticas).

> 📷 **[FIGURA: `P2_variacao_fatorial.png` — barras da % de variação por efeito
> (F1 original): C dominante, B em segundo]**
>
> 📷 **[FIGURA: `variacao_f1logit.png` — % de variação por efeito (F1 logit):
> realça a interação BC]**
>
> 📷 **[FIGURA: `variacao_acc.png` — % de variação por efeito (Acurácia original)
> *(opcional)*]**
>
> 📷 **[FIGURA: `variacao_acclogit.png` — % de variação por efeito (Acurácia logit)
> *(opcional)*]**

**Modelo de regressão a partir dos q (F1 logit).** Com as variáveis codificadas em
−1/+1, o modelo do fatorial é:

$$\hat{y} = q_0 + q_B x_B + q_C x_C + q_D x_D + q_{BC}x_Bx_C + q_{BD}x_Bx_D + q_{CD}x_Cx_D + q_{BCD}x_Bx_Cx_D$$

$$\hat{y}_{\text{logit(F1)}} = 1{,}194 - 0{,}483\,x_B + 0{,}900\,x_C + 0{,}040\,x_D - 0{,}253\,x_Bx_C + 0{,}094\,x_Bx_D + 0{,}010\,x_Cx_D - 0{,}002\,x_Bx_Cx_D$$

*(Acurácia logit: $\hat{y} = 1{,}192 - 0{,}488\,x_B + 0{,}894\,x_C - 0{,}023\,x_D - 0{,}257\,x_Bx_C + 0{,}031\,x_Bx_D + 0{,}001\,x_Cx_D - 0{,}011\,x_Bx_Cx_D$.)*
O valor previsto de qualquer tratamento sai trocando $x_\bullet$ por seus sinais ±1
(ex.: C unânime no MovieReview, texto curto → $x_B{=}{-}1, x_C{=}{+}1, x_D{=}{-}1$).

**Decomposição da variação (SS), F1 logit.** $SS_i = N\,q_i^2$ (N = 40); o erro é
$SSE=\sum e_{ij}^2$; e $SST = \sum_i SS_i + SSE = \sum(y-\bar{y})^2$:

| Componente | SS | g.l. | % da variação |
|---|---|---|---|
| C — Consenso | 32,3928 | 1 | 72,36 |
| B — Domínio | 9,3442 | 1 | 20,87 |
| BC | 2,5547 | 1 | 5,71 |
| BD | 0,3500 | 1 | 0,78 |
| D — Tamanho | 0,0651 | 1 | 0,15 |
| CD | 0,0044 | 1 | 0,01 |
| BCD | 0,0001 | 1 | 0,00 |
| **Erro (SSE)** | **0,0541** | **32** | **0,12** |
| **Total (SST)** | **44,7654** | **39** | **100,00** |

*(Acurácia logit, análogo: SST = 44,2271; SSE = 0,0490 (g.l. 32); C = 31,9611
(72,27%); B = 9,5149 (21,51%); BC = 2,6381 (5,97%); demais < 0,1%.)* A soma dos SS
dos 7 efeitos mais o SSE reconstitui o SST — toda a variação fica alocada, e o
**Erro responde por só 0,12%**.

### 2.3 Comparação original × logit (% de variação, F1)

| Efeito | F1 original | logit(F1) |
|---|---|---|
| C (Consenso) | 81,9% | 72,4% |
| B (Domínio) | 15,9% | 20,9% |
| BC | 0,3% | 5,7% |
| BD | 1,2% | 0,8% |

A ordem dos fatores se mantém (Consenso ≫ Domínio). A transformação **realça a
interação BC** e, importante, **derruba o BCD para não-significativo** — no espaço
original ele só "aparecia" significativo por causa das premissas violadas.

### 2.4 Contrastes de efeitos (IC 95%)

Um **contraste** é uma combinação linear dos efeitos $q_i$ cujos coeficientes
**somam zero** ($\sum h_i = 0$), usada para comparar efeitos entre si. Variância e IC
seguem Jain: $u=\sum_i h_i q_i$, $s^2_u=\dfrac{s_e^2\sum_i h_i^2}{2^k r}$,
$\text{IC}=u\pm t_{[0{,}975;\,32]}\,s_u$. Escolhidos pela leitura dos efeitos
(C ≫ |B| ≫ |BC|), na escala logit:

| Contraste | $u=\sum h_i q_i$ (F1) | IC 95% (F1) | $u$ (Acur.) | IC 95% (Acur.) | Signif. |
|---|---|---|---|---|---|
| **C − B** (consenso vs domínio) | +1,383 | [+1,365; +1,402] | +1,382 | [+1,364; +1,399] | ✓ |
| **BC − BD** (maiores interações) | −0,346 | [−0,365; −0,328] | −0,288 | [−0,305; −0,270] | ✓ |
| **principais − interações** (médias) | +0,190 | [+0,180; +0,200] | +0,187 | [+0,177; +0,197] | ✓ |

*(Contrastes sobre os coeficientes $q$ de Jain; nenhum IC contém 0 ⇒ todas as
diferenças são significativas.)*

- **C − B ≫ 0**: o efeito do **consenso** é significativamente maior (e de sinal
  oposto, positivo) que o do **domínio** — confirma C como fator dominante, agora por
  um teste formal *entre* efeitos, não só pela % de variação.
- **BC − BD < 0** (IC exclui 0): a interação **B×C é significativamente mais forte**
  que B×D — entre as interações, é a BC que importa.
- **principais − interações > 0**: na média, os **efeitos principais superam as
  interações** com 95% de confiança — o sistema é governado por efeitos de 1ª ordem.

### 2.5 Interpretação (P2)

- **O grau de consenso é o fator dominante** (~72% da variação): exigir
  **unanimidade (3×0)** eleva fortemente a qualidade frente à maioria simples (2×1).
- **O domínio é o segundo fator** (~21%): o domínio técnico (Books) reduz a
  qualidade frente ao geral (MovieReview).
- A **interação BC** (~6%) indica que o ganho da unanimidade depende do domínio.
- **Tamanho do texto (D)** e as demais interações são desprezíveis; **CD não é
  significativo**. Como o erro é ínfimo, a leitura relevante é o **tamanho do
  efeito e a % de variação**, não a significância isolada (com erro tão baixo,
  quase todo efeito sai "significativo" pelo IC).
- ⚠️ **Ressalva de confundimento:** B contrasta MovieReview (2 classes) com Books
  (8 classes) — o efeito do "domínio" mistura domínio e nº de classes.

---

## 3. Eixo 2 — Comparação de Sistemas (P1)

**Desenho.** Para cada dataset, compara-se **por fold** o **F1-macro** do RoBERTa
treinado com **voto majoritário** vs. com **dados reais (ground truth)**, na mesma
partição k-fold do benchmark. Análise por **intervalos de confiança (IC 95%)** da
diferença pareada + **correção de múltiplas comparações** (Bonferroni e BH) sobre 4
testes (um por dataset). Convenção: `diff = real − majoritário` (positivo ⇒ reais
melhores).

### 3.1 Médias e IC da diferença

| Dataset | F1 Majoritário | F1 Real | Diferença (IC 95%) | Conclusão |
|---|---|---|---|---|
| **MovieReview** | 0,888 | 0,890 | **+0,003 [−0,005; +0,010]** | **Equivalentes** (IC contém 0) |
| AGNews | 0,871 | 0,917 | +0,046 [+0,040; +0,052] | Real melhor |
| Books | 0,728 | 0,872 | +0,144 [+0,136; +0,152] | Real melhor |
| DBLP | 0,614 | 0,814 | +0,201 [+0,193; +0,208] | Real melhor |

Agregado (35 folds): diferença média **+0,106 [+0,077; +0,134]**.

> 📷 **[FIGURA: `P1_forest_plot_diferenca.png` — forest plot dos IC 95% da diferença
> (real − majoritário) por dataset, com a linha do zero: MovieReview cruza o zero,
> os demais não]**

### 3.2 Correção de múltiplas comparações (t pareado por dataset)

| Dataset | p bruto | p Bonferroni | p BH | Signif. (Bonf. / BH) |
|---|---|---|---|---|
| **MovieReview** | 0,444 | 1,000 | 0,444 | **Não / Não** |
| AGNews | 3,1e-05 | 1,2e-04 | 4,1e-05 | Sim / Sim |
| Books | 1,5e-11 | 5,9e-11 | 3,0e-11 | Sim / Sim |
| DBLP | 7,2e-13 | 2,9e-12 | 2,9e-12 | Sim / Sim |

### 3.3 Interpretação (P1)

- **MovieReview**: as três evidências (IC contendo 0, Bonferroni e BH
  não-significativos) **convergem** → o RoBERTa treinado com rótulos do **voto
  majoritário é estatisticamente equivalente** ao treinado com *ground truth*. A
  anotação automática sai praticamente "de graça" nesse domínio.
- **AGNews, Books, DBLP**: diferença **significativa** mesmo após correção → ainda
  há perda relevante (pior no DBLP, +0,20 de F1). Domínios técnicos/multiclasse
  sofrem mais com o ruído do consenso.
- Bonferroni e BH dão a **mesma decisão** (os p significativos são ~1e-11 e o único
  não-significativo, 0,44, está longe de 0,05) — reforça a robustez.

---

## 4. Eixo 3 (complementar) — Confiança × Qualidade (P3, regressão)

**Desenho.** Nível da anotação (**N = 208 155** exemplos, 4 datasets; resposta
desbalanceada: 168 079 acertos / 40 076 erros). Modelo de probabilidade linear
(LPM, resposta `correct` ∈ {0,1}), erros-padrão robustos (HC1). Avaliou-se uma
sequência de modelos crescentes:

| Modelo | Preditores | R² | R²_aj | AIC |
|---|---|---|---|---|
| M1 | confiança média | 0,0532 | 0,0532 | 191 896 |
| M2 | confianças individuais (3 LLMs) | 0,0533 | 0,0533 | 191 881 |
| M3 | confiança média + consenso | 0,1117 | 0,1117 | 178 625 |
| M4 | confiança média × consenso | 0,1222 | 0,1222 | 176 135 |
| M5 | + domínio (dataset) | 0,1387 | 0,1387 | 172 206 |
| M6 | confianças + std + consenso + domínio | 0,1390 | 0,1390 | 172 131 |

**O resultado é fraco mesmo no melhor modelo.** A confiança **sozinha** (M1/M2)
explica apenas **~5%** da variação do acerto. O salto de R² vem de **consenso** e
**domínio** — não da confiança: ao adicionar consenso (M1→M3) o R² triplica
(0,05→0,11), e o modelo cheio (M6) chega a só **0,139**. Ou seja, **86% da variação
do acerto permanece inexplicada**.

**Coeficientes (M6, EP HC1):**

| Termo | coef | p | IC 95% |
|---|---|---|---|
| conf_llama | +0,881 | 5e-15 | [+0,661; +1,102] |
| conf_qwen | +0,725 | 3e-04 | [+0,335; +1,114] |
| conf_deepseek | +0,608 | 2e-10 | [+0,420; +0,796] |
| **std_confidence** | **+1,357** | 2e-08 | [+0,884; +1,830] |
| consenso 3×0 | +0,264 | ≈0 | [+0,257; +0,270] |
| dataset = movie | +0,053 | 2e-86 | [+0,048; +0,059] |
| dataset = books | −0,092 | <1e-270 | [−0,097; −0,087] |
| dataset = dblp | −0,160 | ≈0 | [−0,165; −0,155] |

**Diagnósticos do LPM (M6):** Shapiro **W=0,749, p≈2e-76**; Breusch-Pagan
**p≈0**; Durbin-Watson **2,00**.

### 4.1 Premissa de linearidade — **violada**

A regressão linear (OLS/LPM) pressupõe **relação linear** entre o preditor e a
resposta (forma funcional correta). Aqui essa premissa **não se sustenta**. A
verificação foi feita por **dispersão + suavização LOESS** (Passo 5 do notebook) e
confirmada agregando a taxa de acerto por faixa de confiança média:

| Faixa de confiança | Taxa de acerto | % das anotações |
|---|---|---|
| 0,6–0,7 | 0,445 | 0,6% |
| 0,7–0,8 | 0,508 | 2,3% |
| 0,8–0,9 | 0,537 | 3,4% |
| 0,9–0,95 | 0,559 | 3,1% |
| 0,95–0,99 | 0,610 | 7,8% |
| **0,99–1,0** | **0,857** | **82,9%** |

Duas patologias saltam à vista:

1. **Não-linearidade (forma em "L invertido"/degrau):** a taxa de acerto fica
   **quase plana (~0,45 → 0,61)** em toda a faixa 0,6–0,99 e só **dá um salto abrupto
   para 0,86** no último bin. Não é uma reta — é praticamente constante e depois um
   degrau. Uma curva LOESS confirma o formato. Ajustar uma reta a isso é
   **especificação incorreta do modelo**: o coeficiente linear "médio" não descreve a
   relação real em nenhuma faixa.
2. **Saturação do preditor:** **82,9%** das anotações têm confiança **≥ 0,99**
   (mediana = 0,999). Quase não há variância no regressor onde está a massa dos dados,
   então o ajuste linear é dominado por um único ponto de apoio.

Em conjunto, isso explica diretamente o **R² baixíssimo**: não é a resposta binária
em si que limita o ajuste (o LPM acomoda isso), e sim a **ausência de relação linear**
e a **saturação** do preditor.

> 📷 **[FIGURA: `_prem_reg_linearidade.png` — premissa de linearidade (dispersão +
> LOESS) de `correct` × confiança: a curva LOESS é plana e depois sobe, não é reta]**
>
> 📷 **[FIGURA: `P3_acerto_vs_confianca.png` — taxa de acerto × confiança média, UMA
> curva por dataset: confiança saturada perto de 1 e relação fraca/não-linear (o
> "degrau")]**

### 4.2 Demais resultados (por que são ruins)

- **Poder explicativo baixíssimo (R²_aj ≈ 0,05–0,14).** A confiança é, na melhor das
  hipóteses, um **preditor fraco** do acerto; sozinha não chega a 6%.
- **Contradição da hipótese H6** (menor dispersão ⇒ maior qualidade): `std_confidence`
  saiu com coeficiente **positivo (+1,36)** — o **oposto** do esperado. Em vez de
  "modelos concordantes e confiantes acertam mais", o sinal sugere que a dispersão
  entre modelos não funciona como indicador de qualidade no sentido proposto.
- **Premissas fortemente violadas:** normalidade e homoscedasticidade rejeitadas
  (Shapiro/Breusch-Pagan p≈0) — esperado para resposta binária, mas reforça que o
  LPM aqui é descritivo, não um bom modelo preditivo.
- O que **de fato** discrimina o acerto são **consenso** e **domínio** (efeitos
  fortes e estáveis), não a confiança — coerente com os Eixos 1, 2 e 4.

> 📷 **[FIGURA: `_prem_reg_residuos.png` — premissas da regressão (M6): histograma,
> QQ-plot e resíduos×ajustados — normalidade e homoscedasticidade rejeitadas (esperado
> no LPM, daí os EP robustos HC1)]**
>
> 📷 **[FIGURA: `P3b_dispersao_confianca.png` — dispersão/histograma da confiança
> mostrando a concentração perto de 1 *(opcional)*]**

### 4.3 Justificativa da mudança de proposta

A proposta inicial previa usar a **confiança (log-probs) dos LLMs como critério de
filtragem/ponderação** das anotações (descartar/penalizar exemplos de baixa
confiança). **Os resultados acima invalidam essa via:**

1. **A confiança não prediz o acerto** de forma útil (R²_aj ≤ 0,06 sozinha; ≤ 0,14
   mesmo somada a tudo) → filtrar por confiança **não separaria** anotações boas de
   ruins.
2. **Saturação perto de 1** → praticamente todas as anotações teriam "alta
   confiança", tornando o limiar **inócuo** como filtro.
3. **`std_confidence` com sinal invertido** → a dispersão entre modelos também **não**
   serve como sinal de qualidade na direção esperada.

Por isso, **abandonou-se a confiança como mecanismo de filtragem** e adotou-se o
**grau de consenso** como critério de qualidade — decisão **sustentada pelos dados**:
o consenso é o fator dominante no fatorial (~72% da variação, Eixo 1) e, na própria
regressão, é o termo que mais eleva o R² (M1→M3: 0,05→0,11). A confiança fica como
**trabalho futuro** (calibração dos log-probs + modelagem não-linear), não como
filtro do *pipeline* atual.

---

## 5. Eixo 4 — Projeto de 1 Fator (Domínio)

**Desenho.** Isola o **domínio** num projeto de **um fator com 4 níveis**
(datasets), via **ANOVA desbalanceada** (amostras de tamanhos diferentes, $r_j$ por
nível): SSY, SS0, SST, **SSA = Σ rⱼαⱼ²**, SSE; **F vs F-tabela**; ICs dos efeitos
(μ, αⱼ, μ+αⱼ, contrastes); g.l. do erro = N − a. Resposta = **F1-macro** (e
Acurácia) do **consenso por voto majoritário** por réplica.

**Réplicas (pastas com *timestamp*):** MovieReview = 5, Books = 5, **AGNews = 2,
DBLP = 4** → **N = 16**, a = 4, g.l. do erro = 12.

**Tabela de dados (layout Jain — réplicas, soma, média e efeito).** Para cada nível
(dataset) listam-se as réplicas, a soma da linha, a **média da linha** ($\bar{y}_{.j}$)
e o **efeito** $\alpha_j = \bar{y}_{.j}-\mu$ (desvio frente à média geral $\mu$).

**F1-macro** (média geral $\mu = 0{,}779$):

| Nível (domínio) | rⱼ | Réplicas (F1) | Soma | Média ȳ.j | Efeito αⱼ |
|---|---|---|---|---|---|
| MovieReview (geral) | 5 | (0,914; 0,913; 0,912; 0,914; 0,913) | 4,566 | 0,913 | **+0,134** |
| AGNews (geral) | 2 | (0,865; 0,867) | 1,732 | 0,866 | +0,087 |
| Books (técnico) | 5 | (0,737; 0,737; 0,738; 0,738; 0,737) | 3,687 | 0,737 | −0,042 |
| DBLP (técnico) | 4 | (0,620; 0,620; 0,619; 0,620) | 2,479 | 0,620 | **−0,159** |
| **Total** | **16** | — | **12,464** | **μ = 0,779** | (Σαⱼ ponderado = 0) |

**Acurácia** (média geral $\mu = 0{,}782$):

| Nível (domínio) | rⱼ | Réplicas (Acur.) | Soma | Média ȳ.j | Efeito αⱼ |
|---|---|---|---|---|---|
| MovieReview (geral) | 5 | (0,914; 0,913; 0,912; 0,914; 0,913) | 4,566 | 0,913 | **+0,131** |
| AGNews (geral) | 2 | (0,868; 0,869) | 1,737 | 0,868 | +0,087 |
| Books (técnico) | 5 | (0,724; 0,723; 0,724; 0,724; 0,724) | 3,618 | 0,724 | −0,058 |
| DBLP (técnico) | 4 | (0,646; 0,648; 0,648; 0,647) | 2,589 | 0,647 | **−0,135** |
| **Total** | **16** | — | **12,509** | **μ = 0,782** | (Σαⱼ ponderado = 0) |

> Leitura (igual ao exemplo de Jain): *em média, o consenso no MovieReview tem F1
> 0,134 acima da média geral; no DBLP, 0,159 abaixo.* A ordem dos efeitos —
> **MovieReview > AGNews > Books > DBLP** — é a mesma em F1 e Acurácia. (Réplicas
> quase idênticas dentro de cada dataset ⇒ erro experimental ínfimo, ver §5.2.)

### 5.1 Premissas (verificadas antes)

Com os 4 níveis (todos com ≥ 2 réplicas), as premissas são **atendidas no espaço
original** (não foi preciso transformar):

| Resposta | Shapiro p | Levene p | Veredito |
|---|---|---|---|
| F1 | 0,382 | 0,238 | **OK** (normal e homoscedástico) |
| Acurácia | 0,500 | 0,295 | **OK** |

> 📷 **[FIGURA: `_prem_um_fator.png` — diagnósticos das premissas do projeto de 1
> fator (histograma, QQ-plot e resíduos×ajustados de F1 e Acurácia): aqui as
> premissas são atendidas no espaço original (sem precisar de logit)]**

### 5.2 Tabela ANOVA — F1-macro

| Componente | SS | g.l. | MS | % Var |
|---|---|---|---|---|
| y − ȳ.. (SST) | 0,2152 | 15 | — | 100,0 |
| **A — Domínio (SSA)** | 0,2152 | 3 | 0,0717 | **99,997** |
| Erros (SSE) | ~0,0000 | 12 | ~0 | 0,003 |

**F = 156 051** ≫ **F-tabela(5%; 3,12) = 3,49** (p ≈ 8×10⁻²⁸) → **rejeita H₀**: o
domínio afeta significativamente a qualidade. Média geral **μ = 0,779**.

### 5.3 Efeitos e médias por nível (F1, IC 95%)

| Dataset (domínio) | rⱼ | Média (ȳ.j) | Efeito αⱼ | αⱼ IC 95% | 0 ∉ IC |
|---|---|---|---|---|---|
| MovieReview (geral) | 5 | 0,913 | **+0,134** | [+0,134; +0,135] | ✓ |
| AGNews (geral) | 2 | 0,866 | +0,087 | [+0,086; +0,088] | ✓ |
| Books (técnico) | 5 | 0,737 | −0,042 | [−0,042; −0,041] | ✓ |
| DBLP (técnico) | 4 | 0,620 | **−0,159** | [−0,160; −0,158] | ✓ |

**Todos os níveis diferem da média geral** (0 ∉ IC) e, nas comparações por par
(contrastes), **todos os 6 pares diferem** (IC da diferença não contém 0). Ordem:
**MovieReview > AGNews > Books > DBLP**.

### 5.4 Acurácia

Mesmo padrão: **F = 104 346** ≫ 3,49 (p ≈ 9×10⁻²⁷); domínio ≈ **100%** da variação;
médias 0,913 (movie) > 0,868 (agnews) > 0,724 (books) > 0,647 (dblp); todos os
níveis e pares diferem.

### 5.5 Interpretação (Eixo 4)

- O **domínio é um fator altamente significativo** e os **4 datasets diferem entre
  si** em qualidade do consenso, tanto em F1 quanto em acurácia.
- A % de variação do domínio é **~100%** porque o **erro intra-dataset é ínfimo**
  (~0,003%): o consenso é quase determinístico entre réplicas — daí o F astronômico.
  A leitura relevante é a **ordenação e as diferenças entre níveis** (todas
  significativas), não a magnitude do F.
- Confirma e detalha o achado do fatorial (fator B): domínios **gerais**
  (MovieReview, AGNews) entregam consenso de melhor qualidade que os **técnicos**
  (Books, DBLP); o DBLP (10 classes) é o pior.
- ⚠️ **Confundimento domínio × nº de classes** persiste (geral = 2–4 classes;
  técnico = 8–10 classes). *(DBLP com 4 réplicas; uma 5ª, se coletada, apenas
  reforça o desenho.)*

---

## 6. Síntese integrada

- **Anotação automática por voto majoritário é viável em domínio geral**
  (MovieReview equivalente ao *ground truth*); em **domínios técnicos** ainda há
  perda significativa. **(P1)**
- **O grau de consenso é o que mais importa** para a qualidade da anotação
  (~72% da variação no fatorial); exigir **unanimidade** é o melhor critério de
  qualidade/filtragem. O **domínio** é o segundo fator (~21%); tamanho do texto é
  irrelevante. **(P2)**
- A **confiança** prediz o acerto, mas é um **indicador fraco e não-linear**
  (saturada); requer calibração para ser útil como filtro. **(P3)**
- O **projeto de 1 fator** confirma e detalha o efeito do domínio: os **4 datasets
  diferem significativamente** entre si, na ordem **MovieReview > AGNews > Books >
  DBLP** (gerais > técnicos). **(Eixo 4)**
- Coerência entre eixos: o fatorial diz "unanimidade = melhor anotação"; a
  comparação de sistemas diz "no domínio geral, a maioria já basta no *downstream*";
  o 1 fator confirma que o **domínio ordena fortemente** a qualidade do consenso.

## 7. Limitações e trabalho futuro

- **Confundimento domínio × nº de classes** no fator B do fatorial (MovieReview 2
  classes vs Books 8 classes).
- **Erro experimental ínfimo** no fatorial (réplicas quase determinísticas) ⇒
  interpretar por tamanho de efeito/% de variação e pelos contrastes, não pela
  significância isolada.
- **Confiança não-calibrada**; aplicar calibração e/ou modelo não-linear (P3).
- **Erro intra-dataset ínfimo** também no projeto de 1 fator (consenso quase
  determinístico) ⇒ F astronômico; interpretar por ordenação/diferenças entre
  níveis. AGNews (2) e DBLP (4) com menos réplicas que o ideal (5).

---

## Apêndice — Notebooks e artefatos

| Eixo | Notebook | Export (JSON/CSV) |
|---|---|---|
| Fatorial 2³ | `analise_projeto_fatorial_2x2x2.ipynb` | `data/results/mq/fatorial_2x2x2/` |
| Comparação de sistemas | `comparacao_sistemas_finetuning.ipynb` | `data/results/mq/comparacao_sistemas/` |
| Regressão (confiança) | `analise_regressao_confianca.ipynb` | `data/results/mq/regressao/` |
| 1 fator (domínio) | `analise_projeto_um_fator_dominio.ipynb` | `data/results/mq/um_fator_dominio/` |

Figuras (tema UFMG) em `src/notebooks/mq/figuras/`.
