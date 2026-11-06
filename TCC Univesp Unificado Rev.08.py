# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:45:50 2026

@author: jefferson.santos
"""

# -*- coding: utf-8 -*-

# =============================================================================
# =============================================================================
# PROJETO: ENEM 2024 - SÃO PAULO
#
# COMPARAÇÃO ENTRE:
#   1. OLS
#   2. OLS com erros-padrão clusterizados por escola
#   3. HLM2 - Modelo Nulo
#   4. HLM2 - Intercepto Aleatório
#   5. HLM2 - Intercepto + Inclinação Aleatórios
#
# NÍVEL 1: ESTUDANTES
# NÍVEL 2: ESCOLAS
#
# Autor: Jefferson Silva Pereira dos Santos
# =============================================================================
# =============================================================================


# =============================================================================
# 0. BIBLIOTECAS
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.anova import anova_lm


warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['figure.dpi'] = 120


# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================

ARQUIVO_PARTICIPANTES = 'PARTICIPANTES_2024.csv'
ARQUIVO_RESULTADOS = 'RESULTADOS_2024.csv'

UF_ANALISE = 'SP'

MIN_ALUNOS_ESCOLA = 7

PASTA_RESULTADOS = 'RESULTADOS_MODELAGEM_ENEM_2024'

os.makedirs(PASTA_RESULTADOS, exist_ok=True)


# =============================================================================
# 2. FUNÇÃO AUXILIAR PARA IMPRIMIR TÍTULOS
# =============================================================================

def titulo(texto):
    print("\n")
    print("=" * 90)
    print(texto)
    print("=" * 90)


# =============================================================================
# 3. IMPORTAÇÃO DOS PARTICIPANTES
# =============================================================================

titulo("3. IMPORTAÇÃO DA BASE DE PARTICIPANTES")

print("Carregando dataframe de participantes...")

df_participantes = pd.read_csv(
    ARQUIVO_PARTICIPANTES,
    sep=';',
    encoding='latin1'
)

print(
    f"Total de participantes carregados: "
    f"{len(df_participantes):,}"
)


# =============================================================================
# 4. FILTRO PARA SÃO PAULO
# =============================================================================

df_participantes_sp = df_participantes[
    df_participantes['SG_UF_PROVA'] == UF_ANALISE
].copy()

print(
    f"Participantes em {UF_ANALISE}: "
    f"{len(df_participantes_sp):,}"
)


# =============================================================================
# 5. IMPORTAÇÃO DOS RESULTADOS
# =============================================================================

titulo("5. IMPORTAÇÃO DA BASE DE RESULTADOS")

colunas_resultados = [
    'CO_ESCOLA',
    'CO_MUNICIPIO_ESC',
    'NU_NOTA_CN',
    'NU_NOTA_CH',
    'NU_NOTA_LC',
    'NU_NOTA_MT',
    'NU_NOTA_REDACAO'
]

print("Carregando resultados...")

df_resultados = pd.read_csv(
    ARQUIVO_RESULTADOS,
    sep=';',
    encoding='latin1',
    usecols=colunas_resultados
)

print(
    f"Resultados carregados: "
    f"{len(df_resultados):,}"
)


# =============================================================================
# 6. ALINHAMENTO ENTRE PARTICIPANTES E RESULTADOS
# =============================================================================

titulo("6. ALINHAMENTO DAS BASES")

print(
    "Realizando alinhamento das linhas conforme a estrutura "
    "dos microdados..."
)

df_resultados_sp = df_resultados.iloc[
    df_participantes_sp.index
].copy()


# =============================================================================
# 7. CONSOLIDAÇÃO
# =============================================================================

df_enem_sp = pd.concat(
    [
        df_participantes_sp.reset_index(drop=True),
        df_resultados_sp.reset_index(drop=True)
    ],
    axis=1
)

print(
    f"Base consolidada SP: "
    f"{len(df_enem_sp):,} registros"
)


# Liberação de memória
del df_participantes
del df_participantes_sp
del df_resultados
del df_resultados_sp


# =============================================================================
# 8. VARIÁVEL DEPENDENTE
# =============================================================================

titulo("8. CONSTRUÇÃO DA VARIÁVEL DEPENDENTE")

notas_cols = [
    'NU_NOTA_CN',
    'NU_NOTA_CH',
    'NU_NOTA_LC',
    'NU_NOTA_MT',
    'NU_NOTA_REDACAO'
]

print("Observações antes do tratamento:", len(df_enem_sp))


# Apenas alunos com todas as notas disponíveis
df_enem_sp = df_enem_sp.dropna(
    subset=notas_cols
).copy()


# Desempenho médio
df_enem_sp['desempenho_medio'] = (
    df_enem_sp[notas_cols]
    .mean(axis=1)
)


print(
    "Observações após notas completas:",
    f"{len(df_enem_sp):,}"
)


# =============================================================================
# 9. INSPEÇÃO DA VARIÁVEL Q007
# =============================================================================

titulo("9. INSPEÇÃO DA VARIÁVEL Q007")

print("Categorias encontradas em Q007:")

print(
    df_enem_sp['Q007']
    .value_counts(dropna=False)
    .sort_index()
)


# =============================================================================
# 10. TRATAMENTO ORDINAL DA RENDA
# =============================================================================
#
# IMPORTANTE:
#
# NÃO utilizamos pd.factorize().
#
# O factorize() depende da ordem de aparecimento dos dados e não deve ser
# usado como garantia de uma escala ordinal.
#
# O pesquisador deve informar explicitamente o código ordinal.
#
# ---------------------------------------------------------------------------
# ATENÇÃO:
# Confira os códigos abaixo com o DICIONÁRIO OFICIAL DO ENEM 2024.
#
# Se Q007 estiver codificado como A, B, C..., mantenha o exemplo e ajuste
# conforme a documentação oficial.
# =============================================================================


# -----------------------------------------------------------------------------
# EXEMPLO DE ESTRUTURA
# -----------------------------------------------------------------------------

MAPA_RENDA = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16
}


categorias_q007 = (
    df_enem_sp['Q007']
    .dropna()
    .unique()
)


print("\nCategorias detectadas:")

for categoria in sorted(categorias_q007):
    print(repr(categoria))


categorias_sem_mapa = [
    categoria
    for categoria in categorias_q007
    if categoria not in MAPA_RENDA
]


if len(categorias_sem_mapa) > 0:

    print("\n" + "!" * 90)
    print("ATENÇÃO: O MAPA DE Q007 AINDA NÃO FOI CONFIGURADO")
    print("!" * 90)

    print("\nCategorias sem codificação:")
    print(categorias_sem_mapa)

    print(
        "\nConsulte o dicionário oficial do ENEM 2024 e preencha "
        "a variável MAPA_RENDA antes de continuar."
    )

    raise ValueError(
        "MAPA_RENDA não configurado para todas as categorias de Q007."
    )


# Aplicação do mapa
df_enem_sp['renda_fam_num'] = (
    df_enem_sp['Q007']
    .map(MAPA_RENDA)
)


# Verificação
print("\nDistribuição da renda ordinal:")

print(
    df_enem_sp['renda_fam_num']
    .value_counts()
    .sort_index()
)


# =============================================================================
# 11. PADRONIZAÇÃO DA RENDA
# =============================================================================

titulo("11. PADRONIZAÇÃO DA RENDA FAMILIAR")

media_renda = (
    df_enem_sp['renda_fam_num']
    .mean()
)

std_renda = (
    df_enem_sp['renda_fam_num']
    .std()
)


df_enem_sp['renda_fam_std'] = (
    df_enem_sp['renda_fam_num']
    - media_renda
) / std_renda


print(f"Média da renda ordinal: {media_renda:.4f}")
print(f"Desvio-padrão:          {std_renda:.4f}")

print(
    "\nMédia padronizada:",
    df_enem_sp['renda_fam_std'].mean()
)

print(
    "Desvio-padrão padronizado:",
    df_enem_sp['renda_fam_std'].std()
)


# =============================================================================
# 12. TRATAMENTO DO NÍVEL 2
# =============================================================================

titulo("12. CONSTRUÇÃO DO NÍVEL 2 - ESCOLAS")

df_modelo = df_enem_sp[
    [
        'desempenho_medio',
        'renda_fam_std',
        'Q007',
        'renda_fam_num',
        'CO_ESCOLA'
    ]
].dropna().copy()


# Transformação da escola em categoria
df_modelo['CO_ESCOLA'] = (
    df_modelo['CO_ESCOLA']
    .astype('category')
)


print(
    "Observações antes do filtro institucional:",
    f"{len(df_modelo):,}"
)


# =============================================================================
# 13. FILTRO N >= 7
# =============================================================================

titulo(
    f"13. FILTRO INSTITUCIONAL - N >= {MIN_ALUNOS_ESCOLA}"
)


contagem_escolas = (
    df_modelo['CO_ESCOLA']
    .value_counts()
)


print("\nDistribuição do número de alunos por escola:")

print(
    contagem_escolas.describe()
)


escolas_validas = (
    contagem_escolas[
        contagem_escolas >= MIN_ALUNOS_ESCOLA
    ]
    .index
)


df_modelo = df_modelo[
    df_modelo['CO_ESCOLA']
    .isin(escolas_validas)
].copy()


df_modelo['CO_ESCOLA'] = (
    df_modelo['CO_ESCOLA']
    .cat.remove_unused_categories()
)


# =============================================================================
# 14. ESTRUTURA DA AMOSTRA ANALÍTICA
# =============================================================================

titulo("14. AMOSTRA ANALÍTICA FINAL")

N = len(df_modelo)

J = df_modelo[
    'CO_ESCOLA'
].nunique()


print(f"Número de estudantes: {N:,}")
print(f"Número de escolas:    {J:,}")


print("\nNúmero de alunos por escola:")

print(
    df_modelo
    .groupby('CO_ESCOLA')
    .size()
    .describe()
)


# =============================================================================
# 15. VERIFICAÇÃO DA VARIAÇÃO DA RENDA DENTRO DAS ESCOLAS
# =============================================================================

titulo("15. VARIAÇÃO DA RENDA DENTRO DAS ESCOLAS")


variacao_renda_escola = (
    df_modelo
    .groupby('CO_ESCOLA', observed=True)['renda_fam_std']
    .agg(
        n='count',
        categorias='nunique',
        desvio='std'
    )
)


print(
    variacao_renda_escola.describe()
)


escolas_sem_variacao = (
    variacao_renda_escola['categorias'] <= 1
).sum()


print(
    "\nEscolas sem variação da renda:",
    escolas_sem_variacao
)


# =============================================================================
# 16. OLS - MODELO LINEAR
# =============================================================================

titulo("16. MODELO OLS")

formula_ols = (
    'desempenho_medio ~ renda_fam_std'
)


modelo_ols = smf.ols(
    formula=formula_ols,
    data=df_modelo
)


resultado_ols = modelo_ols.fit()


print(
    resultado_ols.summary()
)


# =============================================================================
# 17. DIAGNÓSTICO OLS
# =============================================================================

titulo("17. DIAGNÓSTICO DOS RESÍDUOS - OLS")


residuos_ols = (
    resultado_ols.resid
)

ajustados_ols = (
    resultado_ols.fittedvalues
)


# -----------------------------------------------------------------------------
# Jarque-Bera
# -----------------------------------------------------------------------------

jb_stat, jb_pvalor, skew, kurtosis = (
    jarque_bera(residuos_ols)
)


print(
    f"Jarque-Bera: {jb_stat:.4f}"
)

print(
    f"p-valor:     {jb_pvalor:.6f}"
)


# -----------------------------------------------------------------------------
# Breusch-Pagan
# -----------------------------------------------------------------------------

bp = het_breuschpagan(
    residuos_ols,
    resultado_ols.model.exog
)


print(
    f"\nBreusch-Pagan LM: {bp[0]:.4f}"
)

print(
    f"p-valor:          {bp[1]:.6f}"
)


# =============================================================================
# 18. OLS COM ERROS-PADRÃO CLUSTERIZADOS POR ESCOLA
# =============================================================================

titulo(
    "18. OLS COM ERROS-PADRÃO CLUSTERIZADOS POR ESCOLA"
)


resultado_ols_cluster = (
    resultado_ols.get_robustcov_results(
        cov_type='cluster',
        groups=df_modelo['CO_ESCOLA']
    )
)


print(
    resultado_ols_cluster.summary()
)
#%%
# =============================================================================
# GERAÇÃO DOS GRÁFICOS DIAGNÓSTICOS DO MODELO OLS
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import os

# Configuração de fontes acadêmicas padrão
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['figure.dpi'] = 120

# -----------------------------------------------------------------------------
# GRÁFICO 1: Relação entre Renda Familiar e Desempenho Médio (OLS)
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 6))

sns.regplot(
    data=df_modelo,
    x='renda_fam_std',
    y='desempenho_medio',
    scatter_kws={'alpha': 0.25, 'color': 'steelblue'},
    line_kws={'linewidth': 2, 'color': 'darkred'}
)

plt.xlabel('Renda Familiar Padronizada (Z-score)', fontsize=12)
plt.ylabel('Desempenho Médio no ENEM', fontsize=12)
plt.title('Dispersão e Reta de Regressão OLS: Renda vs. Desempenho (SP)', fontsize=13, weight='bold')

plt.tight_layout()
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# GRÁFICO 2: Q-Q Plot Normal dos Resíduos do OLS
# -----------------------------------------------------------------------------
residuos_limpos = np.array(residuos_ols)
residuos_limpos = residuos_limpos[~np.isnan(residuos_limpos)]

fig, ax = plt.subplots(figsize=(8, 7))

fig = sm.qqplot(
    residuos_limpos,
    line='s',
    ax=ax,
    alpha=0.3,
    color='steelblue'
)

ax.set_title('Q-Q Plot Normal dos Resíduos - Modelo OLS (SP)', fontsize=13, weight='bold')
ax.set_xlabel('Quantis Teóricos', fontsize=12)
ax.set_ylabel('Quantis Amostrais (Resíduos)', fontsize=12)

plt.tight_layout()
plt.grid(True)
plt.show()

#%%
# =============================================================================
# 19. HLM2 NULO
# =============================================================================

titulo("19. HLM2 NULO - EMPTY MODEL")


print("Estimando HLM2 nulo...")


modelo_hlm_nulo = sm.MixedLM.from_formula(
    formula='desempenho_medio ~ 1',
    groups='CO_ESCOLA',
    re_formula='1',
    data=df_modelo
)


resultado_hlm_nulo = (
    modelo_hlm_nulo.fit(
        method='powell',
        maxiter=2000,
        reml=True
    )
)


print(
    resultado_hlm_nulo.summary()
)


# =============================================================================
# 20. ICC
# =============================================================================

titulo("20. COEFICIENTE DE CORRELAÇÃO INTRACLASSE - ICC")


var_escola_nulo = float(
    resultado_hlm_nulo.cov_re.iloc[0, 0]
)


var_residual_nulo = (
    resultado_hlm_nulo.scale
)


icc = (
    var_escola_nulo /
    (
        var_escola_nulo
        + var_residual_nulo
    )
)


print(
    f"Variância entre escolas: "
    f"{var_escola_nulo:.6f}"
)

print(
    f"Variância residual:       "
    f"{var_residual_nulo:.6f}"
)

print(
    f"\nICC: {icc:.6f}"
)

print(
    f"ICC (%): {icc*100:.2f}%"
)


# =============================================================================
# 21. HLM2 - INTERCEPTO ALEATÓRIO
# =============================================================================

titulo(
    "21. HLM2 - INTERCEPTO ALEATÓRIO"
)


modelo_hlm_intercepto = sm.MixedLM.from_formula(
    formula='desempenho_medio ~ renda_fam_std',
    groups='CO_ESCOLA',
    re_formula='1',
    data=df_modelo
)


resultado_hlm_intercepto = (
    modelo_hlm_intercepto.fit(
        method='lbfgs',
        maxiter=2000,
        reml=True
    )
)


print(
    resultado_hlm_intercepto.summary()
)


# =============================================================================
# 22. VERIFICAÇÃO DO HLM2
# =============================================================================

titulo(
    "22. DIAGNÓSTICO DO HLM2 - INTERCEPTO"
)


print(
    "Convergiu:",
    resultado_hlm_intercepto.converged
)


print(
    "Log-Likelihood:",
    resultado_hlm_intercepto.llf
)


print(
    "Variância entre escolas:",
    resultado_hlm_intercepto.cov_re.iloc[0, 0]
)


# =============================================================================
# 23. HLM2 - RANDOM SLOPE
# =============================================================================

titulo(
    "23. HLM2 - INTERCEPTO + INCLINAÇÃO ALEATÓRIOS"
)


modelo_hlm_rs = sm.MixedLM.from_formula(
    formula='desempenho_medio ~ renda_fam_std',
    groups='CO_ESCOLA',
    re_formula='1 + renda_fam_std',
    data=df_modelo
)


try:

    resultado_hlm_rs = (
        modelo_hlm_rs.fit(
            method='lbfgs',
            maxiter=5000,
            reml=True
        )
    )

    print(
        resultado_hlm_rs.summary()
    )

except Exception as erro:

    resultado_hlm_rs = None

    print(
        "\nERRO NA ESTIMAÇÃO DO RANDOM SLOPE:"
    )

    print(erro)


# =============================================================================
# 24. DIAGNÓSTICO RANDOM SLOPE
# =============================================================================

if resultado_hlm_rs is not None:

    titulo(
        "24. DIAGNÓSTICO DO RANDOM SLOPE"
    )

    print(
        "Convergiu:",
        resultado_hlm_rs.converged
    )

    print(
        "Log-Likelihood:",
        resultado_hlm_rs.llf
    )

    print(
        "\nMatriz de covariância dos efeitos aleatórios:"
    )

    print(
        resultado_hlm_rs.cov_re
    )


# =============================================================================
# 25. TESTE DE RAZÃO DE VEROSSIMILHANÇA
# =============================================================================
#
# Para comparar:
#
# HLM2 intercepto
# versus
# HLM2 random slope
#
# usamos ML, e NÃO REML, para comparação de modelos com diferentes
# estruturas de efeitos aleatórios.
# =============================================================================


titulo(
    "25. TESTE DE RAZÃO DE VEROSSIMILHANÇA"
)


modelo_hlm_intercepto_ml = (
    sm.MixedLM.from_formula(
        formula='desempenho_medio ~ renda_fam_std',
        groups='CO_ESCOLA',
        re_formula='1',
        data=df_modelo
    )
)


resultado_hlm_intercepto_ml = (
    modelo_hlm_intercepto_ml.fit(
        method='lbfgs',
        maxiter=3000,
        reml=False
    )
)


modelo_hlm_rs_ml = (
    sm.MixedLM.from_formula(
        formula='desempenho_medio ~ renda_fam_std',
        groups='CO_ESCOLA',
        re_formula='1 + renda_fam_std',
        data=df_modelo
    )
)


try:

    resultado_hlm_rs_ml = (
        modelo_hlm_rs_ml.fit(
            method='lbfgs',
            maxiter=5000,
            reml=False
        )
    )


    LR = (
        2 *
        (
            resultado_hlm_rs_ml.llf
            -
            resultado_hlm_intercepto_ml.llf
        )
    )


    diferenca_parametros = (
        len(resultado_hlm_rs_ml.params)
        -
        len(resultado_hlm_intercepto_ml.params)
    )


    pvalor_lr = stats.chi2.sf(
        LR,
        diferenca_parametros
    )


    print(
        f"LogLik HLM2 intercepto: "
        f"{resultado_hlm_intercepto_ml.llf:.4f}"
    )

    print(
        f"LogLik HLM2 random slope: "
        f"{resultado_hlm_rs_ml.llf:.4f}"
    )

    print(
        f"\nLR: {LR:.4f}"
    )

    print(
        f"Graus de liberdade: "
        f"{diferenca_parametros}"
    )

    print(
        f"p-valor: {pvalor_lr:.6f}"
    )


except Exception as erro:

    resultado_hlm_rs_ml = None

    print(
        "Não foi possível estimar o random slope via ML:"
    )

    print(erro)


# =============================================================================
# 26. COMPARAÇÃO OLS × HLM2
# =============================================================================

titulo(
    "26. COMPARAÇÃO DOS MODELOS"
)


linhas_comparacao = []


# -----------------------------------------------------------------------------
# OLS
# -----------------------------------------------------------------------------

linhas_comparacao.append({
    'Modelo': 'OLS',
    'N': resultado_ols.nobs,
    'Escolas': J,
    'Coef_Renda': resultado_ols.params['renda_fam_std'],
    'EP_Renda': resultado_ols.bse['renda_fam_std'],
    'p_Renda': resultado_ols.pvalues['renda_fam_std'],
    'LogLik': resultado_ols.llf,
    'AIC': resultado_ols.aic,
    'BIC': resultado_ols.bic
})


# -----------------------------------------------------------------------------
# OLS Clusterizado
# -----------------------------------------------------------------------------

linhas_comparacao.append({
    'Modelo': 'OLS Clusterizado',
    'N': resultado_ols.nobs,
    'Escolas': J,
    'Coef_Renda': resultado_ols_cluster.params[
        list(resultado_ols.params.index).index(
            'renda_fam_std'
        )
    ],
    'EP_Renda': resultado_ols_cluster.bse[
        list(resultado_ols.params.index).index(
            'renda_fam_std'
        )
    ],
    'p_Renda': resultado_ols_cluster.pvalues[
        list(resultado_ols.params.index).index(
            'renda_fam_std'
        )
    ],
    'LogLik': resultado_ols.llf,
    'AIC': resultado_ols.aic,
    'BIC': resultado_ols.bic
})


# -----------------------------------------------------------------------------
# HLM2 intercepto
# -----------------------------------------------------------------------------

linhas_comparacao.append({
    'Modelo': 'HLM2 Intercepto',
    'N': resultado_hlm_intercepto.nobs,
    'Escolas': J,
    'Coef_Renda': resultado_hlm_intercepto.params[
        'renda_fam_std'
    ],
    'EP_Renda': resultado_hlm_intercepto.bse[
        'renda_fam_std'
    ],
    'p_Renda': resultado_hlm_intercepto.pvalues[
        'renda_fam_std'
    ],
    'LogLik': resultado_hlm_intercepto.llf,
    'AIC': resultado_hlm_intercepto.aic,
    'BIC': resultado_hlm_intercepto.bic
})


# -----------------------------------------------------------------------------
# HLM2 Random Slope
# -----------------------------------------------------------------------------

if resultado_hlm_rs is not None:

    linhas_comparacao.append({
        'Modelo': 'HLM2 Random Slope',
        'N': resultado_hlm_rs.nobs,
        'Escolas': J,
        'Coef_Renda': resultado_hlm_rs.params[
            'renda_fam_std'
        ],
        'EP_Renda': resultado_hlm_rs.bse[
            'renda_fam_std'
        ],
        'p_Renda': resultado_hlm_rs.pvalues[
            'renda_fam_std'
        ],
        'LogLik': resultado_hlm_rs.llf,
        'AIC': resultado_hlm_rs.aic,
        'BIC': resultado_hlm_rs.bic
    })


tabela_comparacao = pd.DataFrame(
    linhas_comparacao
)


print(
    tabela_comparacao.to_string(
        index=False
    )
)


# =============================================================================
# 27. IC 95% DO COEFICIENTE DA RENDA
# =============================================================================

titulo(
    "27. INTERVALOS DE CONFIANÇA - COEFICIENTE DA RENDA"
)


print("\nOLS:")

print(
    resultado_ols.conf_int().loc[
        'renda_fam_std'
    ]
)


print("\nHLM2 Intercepto:")

print(
    resultado_hlm_intercepto.conf_int().loc[
        'renda_fam_std'
    ]
)


if resultado_hlm_rs is not None:

    print("\nHLM2 Random Slope:")

    print(
        resultado_hlm_rs.conf_int().loc[
            'renda_fam_std'
        ]
    )


# =============================================================================
# 28. VARIÂNCIAS DO HLM2 RANDOM SLOPE
# =============================================================================

if resultado_hlm_rs is not None:

    titulo(
        "28. COMPONENTES DE VARIÂNCIA - RANDOM SLOPE"
    )


    cov_re = resultado_hlm_rs.cov_re


    var_intercepto = cov_re.iloc[0, 0]

    var_slope = cov_re.iloc[1, 1]

    cov_intercepto_slope = cov_re.iloc[0, 1]


    print(
        f"Variância do intercepto: "
        f"{var_intercepto:.6f}"
    )

    print(
        f"Variância do slope da renda: "
        f"{var_slope:.6f}"
    )

    print(
        f"Covariância intercepto-slope: "
        f"{cov_intercepto_slope:.6f}"
    )


# =============================================================================
# 29. GRÁFICO 1 - DISTRIBUIÇÃO DO DESEMPENHO
# =============================================================================

titulo(
    "29. GRÁFICO - DISTRIBUIÇÃO DO DESEMPENHO"
)


plt.figure(figsize=(9, 6))

sns.histplot(
    df_modelo['desempenho_medio'],
    bins=30,
    kde=True
)

plt.xlabel(
    'Desempenho médio'
)

plt.ylabel(
    'Frequência'
)

plt.title(
    'Distribuição do desempenho médio no ENEM 2024 - SP'
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        '01_distribuicao_desempenho.png'
    ),
    dpi=300
)

plt.show()


# =============================================================================
# 30. GRÁFICO 2 - RENDA × DESEMPENHO
# =============================================================================

plt.figure(figsize=(9, 6))

sns.regplot(
    data=df_modelo,
    x='renda_fam_std',
    y='desempenho_medio',
    scatter_kws={'alpha': 0.25},
    line_kws={'linewidth': 2}
)

plt.xlabel(
    'Renda familiar padronizada'
)

plt.ylabel(
    'Desempenho médio'
)

plt.title(
    'Relação entre renda familiar e desempenho médio'
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        '02_renda_desempenho_OLS.png'
    ),
    dpi=300
)

plt.show()


# =============================================================================
# 31. GRÁFICO 3 - RESÍDUOS OLS
# =============================================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=ajustados_ols,
    y=residuos_ols,
    alpha=0.35
)

plt.axhline(
    0,
    linestyle='--'
)

plt.xlabel(
    'Valores ajustados'
)

plt.ylabel(
    'Resíduos'
)

plt.title(
    'Resíduos versus valores ajustados - OLS'
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        '03_residuos_OLS.png'
    ),
    dpi=300
)

plt.show()


# =============================================================================
# 32. GRÁFICO 4 - Q-Q PLOT
# =============================================================================

fig = plt.figure(
    figsize=(8, 7)
)

ax = fig.add_subplot(111)

sm.qqplot(
    residuos_ols,
    line='s',
    ax=ax
)

ax.set_title(
    'Q-Q Plot dos resíduos - OLS'
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PASTA_RESULTADOS,
        '04_QQplot_OLS.png'
    ),
    dpi=300
)

plt.show()


# =============================================================================
# 33. EXPORTAÇÃO DA TABELA DE COMPARAÇÃO
# =============================================================================

titulo(
    "33. EXPORTAÇÃO DOS RESULTADOS"
)


arquivo_comparacao = os.path.join(
    PASTA_RESULTADOS,
    'comparacao_modelos.xlsx'
)


tabela_comparacao.to_excel(
    arquivo_comparacao,
    index=False
)


print(
    "Tabela salva em:",
    arquivo_comparacao
)


# =============================================================================
# 34. EXPORTAÇÃO DA BASE ANALÍTICA
# =============================================================================

arquivo_base = os.path.join(
    PASTA_RESULTADOS,
    'base_analitica_839_estudantes.xlsx'
)


df_modelo.to_excel(
    arquivo_base,
    index=False
)


print(
    "Base analítica salva em:",
    arquivo_base
)


# =============================================================================
# 35. RESUMO FINAL
# =============================================================================

titulo(
    "35. RESUMO FINAL DA MODELAGEM"
)


print(
    f"Número de estudantes: {N:,}"
)

print(
    f"Número de escolas:    {J:,}"
)

print(
    f"Filtro institucional: N >= {MIN_ALUNOS_ESCOLA}"
)

print(
    f"\nICC do modelo nulo: "
    f"{icc:.4f} ({icc*100:.2f}%)"
)


print(
    "\nCoeficiente da renda:"
)

print(
    f"OLS:                "
    f"{resultado_ols.params['renda_fam_std']:.4f}"
)

print(
    f"OLS clusterizado:   "
    f"{resultado_ols_cluster.params[1]:.4f}"
)

print(
    f"HLM2 intercepto:    "
    f"{resultado_hlm_intercepto.params['renda_fam_std']:.4f}"
)


if resultado_hlm_rs is not None:

    print(
        f"HLM2 random slope:  "
        f"{resultado_hlm_rs.params['renda_fam_std']:.4f}"
    )


print(
    "\nConvergência HLM2 intercepto:",
    resultado_hlm_intercepto.converged
)


if resultado_hlm_rs is not None:

    print(
        "Convergência HLM2 random slope:",
        resultado_hlm_rs.converged
    )


print(
    "\nArquivos gerados na pasta:"
)

print(
    PASTA_RESULTADOS
)

print(
    "\n✓ SCRIPT CONCLUÍDO."
)
#%%

# -*- coding: utf-8 -*-
"""
HLM2 - ENEM 2024 - São Paulo
Etapa:
    1. HLM2 nulo
    2. HLM2 com renda - intercepto aleatório
    3. HLM2 com renda - random slope
    4. Comparação ML
    5. ICC
    6. Testes LR
    7. Diagnóstico dos modelos

Base:
    df_modelo

Nível 1:
    Estudante

Nível 2:
    Escola (CO_ESCOLA)

Variável dependente:
    desempenho_medio

Variável explicativa:
    renda_fam_std
"""

#%% =============================================================================
# 1. BIBLIOTECAS
# =============================================================================

import warnings

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import chi2


#%% =============================================================================
# 2. VERIFICAÇÃO DA BASE
# =============================================================================

print("=" * 80)
print("VERIFICAÇÃO DA BASE")
print("=" * 80)

print("Dimensão da base:", df_modelo.shape)

print("\nColunas:")
print(df_modelo.columns.tolist())

print("\nTipos:")
print(df_modelo.dtypes)

print("\nValores ausentes:")
print(df_modelo[
    ["desempenho_medio", "renda_fam_std", "CO_ESCOLA"]
].isna().sum())


#%% =============================================================================
# 3. PREPARAÇÃO DA BASE PARA O HLM
# =============================================================================

df_hlm = df_modelo[
    [
        "desempenho_medio",
        "renda_fam_std",
        "CO_ESCOLA"
    ]
].copy()

# Garantir tipos adequados
df_hlm["desempenho_medio"] = pd.to_numeric(
    df_hlm["desempenho_medio"],
    errors="coerce"
)

df_hlm["renda_fam_std"] = pd.to_numeric(
    df_hlm["renda_fam_std"],
    errors="coerce"
)

# Escola como variável categórica/grupo
df_hlm["CO_ESCOLA"] = df_hlm["CO_ESCOLA"].astype(str)

# Remover observações incompletas
df_hlm = df_hlm.dropna().copy()

# Ordenar por escola
df_hlm = df_hlm.sort_values("CO_ESCOLA").reset_index(drop=True)


print("=" * 80)
print("BASE UTILIZADA NO HLM")
print("=" * 80)

print("N de estudantes:", len(df_hlm))

print("N de escolas:",
      df_hlm["CO_ESCOLA"].nunique())

print("\nTamanho das escolas:")

tamanho_escolas = (
    df_hlm
    .groupby("CO_ESCOLA")
    .size()
)

print(tamanho_escolas.describe())


#%% =============================================================================
# 4. FUNÇÃO PARA AJUSTAR MODELOS MIXEDLM
# =============================================================================

def ajustar_mixedlm(formula, re_formula, dados, nome_modelo):
    """
    Tenta estimar o MixedLM usando diferentes otimizadores.

    Critério principal:
        LogLik finito

    Critérios de diagnóstico:
        convergência
        ausência de warnings críticos
        matriz de covariância dos efeitos aleatórios
    """

    print("\n" + "=" * 80)
    print(f"AJUSTANDO: {nome_modelo}")
    print("=" * 80)

    resultados = []

    otimizadores = [
        "bfgs",
        "cg",
        "powell",
        "lbfgs"
    ]

    for metodo in otimizadores:

        print(f"\nTentando otimização: {metodo}")

        with warnings.catch_warnings(record=True) as w:

            warnings.simplefilter("always")

            try:

                modelo = smf.mixedlm(
                    formula=formula,
                    data=dados,
                    groups=dados["CO_ESCOLA"],
                    re_formula=re_formula
                )

                resultado = modelo.fit(
                    reml=False,
                    method=metodo,
                    maxiter=2000,
                    disp=False
                )

                loglik = resultado.llf

                convergiu = getattr(
                    resultado,
                    "converged",
                    False
                )

                n_warnings = len(w)

                # Verificação do LogLik
                loglik_finito = np.isfinite(loglik)

                # Verificação da covariância dos efeitos aleatórios
                try:
                    cov_re = resultado.cov_re

                    autovalores = np.linalg.eigvalsh(
                        np.asarray(cov_re)
                    )

                    cov_re_positiva = np.all(
                        autovalores > 1e-10
                    )

                except Exception:
                    cov_re_positiva = False

                valido = (
                    convergiu
                    and loglik_finito
                )

                print(
                    f"Convergiu: {convergiu} | "
                    f"LogLik: {loglik:.6f} | "
                    f"Warnings: {n_warnings} | "
                    f"Cov RE positiva: {cov_re_positiva}"
                )

                resultados.append({
                    "metodo": metodo,
                    "resultado": resultado,
                    "loglik": loglik,
                    "convergiu": convergiu,
                    "n_warnings": n_warnings,
                    "cov_re_positiva": cov_re_positiva,
                    "valido": valido
                })

            except Exception as e:

                print(
                    f"ERRO no método {metodo}: {e}"
                )

    # Apenas modelos com LogLik finito e convergência
    validos = [
        r for r in resultados
        if r["valido"]
    ]

    if len(validos) == 0:

        raise RuntimeError(
            f"Nenhum ajuste válido encontrado para "
            f"{nome_modelo}."
        )

    # Escolher maior LogLik
    melhor = max(
        validos,
        key=lambda x: x["loglik"]
    )

    print("\n" + "-" * 80)
    print("MELHOR AJUSTE")
    print("-" * 80)

    print("Método:", melhor["metodo"])
    print("Convergiu:", melhor["convergiu"])
    print("LogLik:", melhor["loglik"])
    print("Warnings:", melhor["n_warnings"])

    return melhor["resultado"], resultados


#%% =============================================================================
# 5. HLM2 NULO — RESULTADO JÁ VALIDADO
# =============================================================================

# IMPORTANTE:
# O modelo nulo já foi estimado anteriormente.
#
# Aqui vamos estimá-lo novamente usando a mesma função,
# garantindo que o ML seja comparável aos demais modelos.

formula_nulo = (
    "desempenho_medio ~ 1"
)

hlm2_nulo, tentativas_nulo = ajustar_mixedlm(
    formula=formula_nulo,
    re_formula="1",
    dados=df_hlm,
    nome_modelo="HLM2 NULO"
)


#%% =============================================================================
# 6. RESULTADO DO HLM2 NULO
# =============================================================================

print("\n")
print("=" * 80)
print("HLM2 NULO — ML")
print("=" * 80)

print(hlm2_nulo.summary())


#%% =============================================================================
# 7. COMPONENTES DE VARIÂNCIA — HLM2 NULO
# =============================================================================

tau00_nulo = float(
    hlm2_nulo.cov_re.iloc[0, 0]
)

sigma2_nulo = float(
    hlm2_nulo.scale
)

variancia_total_nulo = (
    tau00_nulo +
    sigma2_nulo
)

icc_nulo = (
    tau00_nulo /
    variancia_total_nulo
)

print("\n" + "=" * 80)
print("COMPONENTES DE VARIÂNCIA — HLM2 NULO")
print("=" * 80)

print(
    f"Variância entre escolas (τ00): "
    f"{tau00_nulo:.6f}"
)

print(
    f"Variância dentro das escolas (σ²): "
    f"{sigma2_nulo:.6f}"
)

print(
    f"Variância total: "
    f"{variancia_total_nulo:.6f}"
)

print(
    f"ICC: "
    f"{icc_nulo:.6f}"
)

print(
    f"ICC (%): "
    f"{icc_nulo * 100:.4f}%"
)


#%% =============================================================================
# 8. HLM2 COM RENDA — INTERCEPTO ALEATÓRIO
# =============================================================================

formula_renda = (
    "desempenho_medio ~ renda_fam_std"
)

hlm2_renda, tentativas_renda = ajustar_mixedlm(
    formula=formula_renda,
    re_formula="1",
    dados=df_hlm,
    nome_modelo="HLM2 — RENDA + INTERCEPTO ALEATÓRIO"
)


#%% =============================================================================
# 9. RESULTADO HLM2 COM RENDA
# =============================================================================

print("\n")
print("=" * 80)
print("HLM2 — RENDA + INTERCEPTO ALEATÓRIO — ML")
print("=" * 80)

print(hlm2_renda.summary())


#%% =============================================================================
# 10. COMPONENTES DE VARIÂNCIA — HLM2 COM RENDA
# =============================================================================

tau00_renda = float(
    hlm2_renda.cov_re.iloc[0, 0]
)

sigma2_renda = float(
    hlm2_renda.scale
)

variancia_total_renda = (
    tau00_renda +
    sigma2_renda
)

icc_renda = (
    tau00_renda /
    variancia_total_renda
)

print("\n" + "=" * 80)
print("COMPONENTES DE VARIÂNCIA — HLM2 COM RENDA")
print("=" * 80)

print(
    f"Variância entre escolas (τ00): "
    f"{tau00_renda:.6f}"
)

print(
    f"Variância dentro das escolas (σ²): "
    f"{sigma2_renda:.6f}"
)

print(
    f"Variância total: "
    f"{variancia_total_renda:.6f}"
)

print(
    f"ICC: "
    f"{icc_renda:.6f}"
)

print(
    f"ICC (%): "
    f"{icc_renda * 100:.4f}%"
)


#%% =============================================================================
# 11. COEFICIENTE DE RENDA
# =============================================================================

print("\n" + "=" * 80)
print("EFEITO DA RENDA")
print("=" * 80)

coef_renda = hlm2_renda.params["renda_fam_std"]
se_renda = hlm2_renda.bse["renda_fam_std"]
p_renda = hlm2_renda.pvalues["renda_fam_std"]

print(
    f"Coeficiente renda: "
    f"{coef_renda:.6f}"
)

print(
    f"Erro-padrão: "
    f"{se_renda:.6f}"
)

print(
    f"p-valor: "
    f"{p_renda:.6f}"
)

print(
    f"IC 95%: "
    f"{hlm2_renda.conf_int().loc['renda_fam_std', 0]:.6f} "
    f"a "
    f"{hlm2_renda.conf_int().loc['renda_fam_std', 1]:.6f}"
)


#%% =============================================================================
# 12. REDUÇÃO DA VARIÂNCIA ENTRE ESCOLAS
# =============================================================================

reducao_tau00 = (
    (tau00_nulo - tau00_renda)
    / tau00_nulo
    * 100
)

print("\n" + "=" * 80)
print("REDUÇÃO DA VARIÂNCIA ENTRE ESCOLAS")
print("=" * 80)

print(
    f"τ00 HLM2 nulo: "
    f"{tau00_nulo:.6f}"
)

print(
    f"τ00 HLM2 + renda: "
    f"{tau00_renda:.6f}"
)

print(
    f"Redução de τ00: "
    f"{reducao_tau00:.4f}%"
)


#%% =============================================================================
# 13. HLM2 COM RANDOM SLOPE
# =============================================================================

formula_slope = (
    "desempenho_medio ~ renda_fam_std"
)

hlm2_slope, tentativas_slope = ajustar_mixedlm(
    formula=formula_slope,
    re_formula="1 + renda_fam_std",
    dados=df_hlm,
    nome_modelo="HLM2 — RENDA + RANDOM SLOPE"
)


#%% =============================================================================
# 14. RESULTADO RANDOM SLOPE
# =============================================================================

print("\n")
print("=" * 80)
print("HLM2 — RENDA + RANDOM SLOPE — ML")
print("=" * 80)

print(hlm2_slope.summary())


#%% =============================================================================
# 15. MATRIZ DE COVARIÂNCIA DOS EFEITOS ALEATÓRIOS
# =============================================================================

print("\n" + "=" * 80)
print("MATRIZ DE COVARIÂNCIA DOS EFEITOS ALEATÓRIOS")
print("=" * 80)

print(
    hlm2_slope.cov_re
)


#%% =============================================================================
# 16. COMPONENTES DO RANDOM SLOPE
# =============================================================================

cov_re_slope = hlm2_slope.cov_re

tau00_slope = float(
    cov_re_slope.loc[
        "Group",
        "Group"
    ]
)

tau01_slope = float(
    cov_re_slope.loc[
        "Group",
        "renda_fam_std"
    ]
)

tau11_slope = float(
    cov_re_slope.loc[
        "renda_fam_std",
        "renda_fam_std"
    ]
)

sigma2_slope = float(
    hlm2_slope.scale
)

variancia_total_slope = (
    tau00_slope +
    sigma2_slope
)

icc_slope = (
    tau00_slope /
    variancia_total_slope
)

print("\n" + "=" * 80)
print("COMPONENTES DO RANDOM SLOPE")
print("=" * 80)

print(
    f"τ00 — variância do intercepto: "
    f"{tau00_slope:.6f}"
)

print(
    f"τ01 — covariância intercepto × renda: "
    f"{tau01_slope:.6f}"
)

print(
    f"τ11 — variância da inclinação da renda: "
    f"{tau11_slope:.6f}"
)

print(
    f"σ² — variância residual: "
    f"{sigma2_slope:.6f}"
)

print(
    f"ICC aproximado: "
    f"{icc_slope:.6f}"
)

print(
    f"ICC aproximado (%): "
    f"{icc_slope * 100:.4f}%"
)


#%% =============================================================================
# 17. EFEITOS FIXOS DO RANDOM SLOPE
# =============================================================================

print("\n" + "=" * 80)
print("EFEITOS FIXOS — RANDOM SLOPE")
print("=" * 80)

print(
    hlm2_slope.params
)

print("\nErros-padrão:")
print(
    hlm2_slope.bse
)

print("\np-valores:")
print(
    hlm2_slope.pvalues
)

print("\nIntervalos de confiança:")
print(
    hlm2_slope.conf_int()
)


#%% =============================================================================
# 18. LOG-LIKELIHOOD DOS MODELOS
# =============================================================================

ll_nulo = hlm2_nulo.llf
ll_renda = hlm2_renda.llf
ll_slope = hlm2_slope.llf

print("\n" + "=" * 80)
print("LOG-LIKELIHOOD")
print("=" * 80)

print(
    f"HLM2 nulo:       {ll_nulo:.6f}"
)

print(
    f"HLM2 + renda:    {ll_renda:.6f}"
)

print(
    f"HLM2 random slope: {ll_slope:.6f}"
)


#%% =============================================================================
# 19. AIC E BIC
# =============================================================================

print("\n" + "=" * 80)
print("AIC E BIC")
print("=" * 80)

print(
    f"HLM2 nulo:"
)

print(
    f"    AIC = {hlm2_nulo.aic:.6f}"
)

print(
    f"    BIC = {hlm2_nulo.bic:.6f}"
)

print(
    f"\nHLM2 + renda:"
)

print(
    f"    AIC = {hlm2_renda.aic:.6f}"
)

print(
    f"    BIC = {hlm2_renda.bic:.6f}"
)

print(
    f"\nHLM2 random slope:"
)

print(
    f"    AIC = {hlm2_slope.aic:.6f}"
)

print(
    f"    BIC = {hlm2_slope.bic:.6f}"
)


#%% =============================================================================
# 20. LR TEST — HLM2 NULO VS HLM2 + RENDA
# =============================================================================

LR_nulo_renda = (
    2 *
    (ll_renda - ll_nulo)
)

df_lr_nulo_renda = 1

p_lr_nulo_renda = chi2.sf(
    LR_nulo_renda,
    df_lr_nulo_renda
)

print("\n" + "=" * 80)
print("TESTE LR — HLM2 NULO VS HLM2 + RENDA")
print("=" * 80)

print(
    f"LogLik nulo: "
    f"{ll_nulo:.6f}"
)

print(
    f"LogLik renda: "
    f"{ll_renda:.6f}"
)

print(
    f"LR = "
    f"{LR_nulo_renda:.6f}"
)

print(
    f"gl = "
    f"{df_lr_nulo_renda}"
)

print(
    f"p-valor = "
    f"{p_lr_nulo_renda:.6f}"
)


#%% =============================================================================
# 21. LR TEST — HLM2 + RENDA VS RANDOM SLOPE
# =============================================================================

LR_renda_slope = (
    2 *
    (ll_slope - ll_renda)
)

df_lr_renda_slope = 2

p_lr_renda_slope = chi2.sf(
    LR_renda_slope,
    df_lr_renda_slope
)

print("\n" + "=" * 80)
print("TESTE LR — INTERCEPTO ALEATÓRIO VS RANDOM SLOPE")
print("=" * 80)

print(
    f"LogLik HLM2 + renda: "
    f"{ll_renda:.6f}"
)

print(
    f"LogLik random slope: "
    f"{ll_slope:.6f}"
)

print(
    f"LR = "
    f"{LR_renda_slope:.6f}"
)

print(
    f"gl = "
    f"{df_lr_renda_slope}"
)

print(
    f"p-valor aproximado = "
    f"{p_lr_renda_slope:.6f}"
)

print("\nATENÇÃO:")
print(
    "O teste LR para inclusão de componentes de variância "
    "possui distribuição não-padrão quando a variância "
    "testada está na fronteira do espaço paramétrico."
)

print(
    "Portanto, o p-valor qui-quadrado deve ser interpretado "
    "como uma aproximação/diagnóstico, e não como um teste "
    "exato da significância da variância do random slope."
)


#%% =============================================================================
# 22. TABELA COMPARATIVA DOS MODELOS
# =============================================================================

tabela_modelos = pd.DataFrame({

    "Modelo": [
        "HLM2 Nulo",
        "HLM2 + Renda",
        "HLM2 + Renda + Random Slope"
    ],

    "N": [
        hlm2_nulo.nobs,
        hlm2_renda.nobs,
        hlm2_slope.nobs
    ],

    "Escolas": [
        df_hlm["CO_ESCOLA"].nunique(),
        df_hlm["CO_ESCOLA"].nunique(),
        df_hlm["CO_ESCOLA"].nunique()
    ],

    "LogLik": [
        hlm2_nulo.llf,
        hlm2_renda.llf,
        hlm2_slope.llf
    ],

    "AIC": [
        hlm2_nulo.aic,
        hlm2_renda.aic,
        hlm2_slope.aic
    ],

    "BIC": [
        hlm2_nulo.bic,
        hlm2_renda.bic,
        hlm2_slope.bic
    ],

    "Variancia_Entre": [
        tau00_nulo,
        tau00_renda,
        tau00_slope
    ],

    "Variancia_Residual": [
        sigma2_nulo,
        sigma2_renda,
        sigma2_slope
    ],

    "ICC": [
        icc_nulo,
        icc_renda,
        icc_slope
    ]

})


print("\n" + "=" * 80)
print("TABELA COMPARATIVA")
print("=" * 80)

print(
    tabela_modelos.to_string(index=False)
)


#%% =============================================================================
# 23. COMPARAÇÃO DA VARIÂNCIA ENTRE ESCOLAS
# =============================================================================

print("\n" + "=" * 80)
print("COMPARAÇÃO DA VARIÂNCIA ENTRE ESCOLAS")
print("=" * 80)

print(
    f"HLM2 Nulo: "
    f"{tau00_nulo:.6f}"
)

print(
    f"HLM2 + Renda: "
    f"{tau00_renda:.6f}"
)

print(
    f"Random Slope: "
    f"{tau00_slope:.6f}"
)

reducao_renda = (
    (tau00_nulo - tau00_renda)
    / tau00_nulo
    * 100
)

reducao_slope = (
    (tau00_nulo - tau00_slope)
    / tau00_nulo
    * 100
)

print(
    f"\nRedução de τ00 após inclusão da renda: "
    f"{reducao_renda:.4f}%"
)

print(
    f"Redução de τ00 no random slope: "
    f"{reducao_slope:.4f}%"
)


#%% =============================================================================
# 24. VARIAÇÃO DA ICC
# =============================================================================

print("\n" + "=" * 80)
print("VARIAÇÃO DO ICC")
print("=" * 80)

print(
    f"ICC HLM2 Nulo: "
    f"{icc_nulo * 100:.4f}%"
)

print(
    f"ICC HLM2 + Renda: "
    f"{icc_renda * 100:.4f}%"
)

print(
    f"ICC Random Slope: "
    f"{icc_slope * 100:.4f}%"
)


#%% =============================================================================
# 25. DIAGNÓSTICO FINAL
# =============================================================================

print("\n" + "=" * 80)
print("DIAGNÓSTICO FINAL DOS MODELOS")
print("=" * 80)

for nome, modelo in [
    ("HLM2 Nulo", hlm2_nulo),
    ("HLM2 + Renda", hlm2_renda),
    ("HLM2 Random Slope", hlm2_slope)
]:

    print("\n" + "-" * 80)
    print(nome)

    print(
        "Convergiu:",
        getattr(modelo, "converged", None)
    )

    print(
        "LogLik finito:",
        np.isfinite(modelo.llf)
    )

    print(
        "LogLik:",
        modelo.llf
    )

    print(
        "AIC:",
        modelo.aic
    )

    print(
        "BIC:",
        modelo.bic
    )

    print(
        "Warnings durante ajuste:",
        "ver tentativas individuais"
    )


#%% =============================================================================
# 26. SALVAR TABELA COMPARATIVA
# =============================================================================

tabela_modelos.to_excel(
    "comparacao_HLM2_ML_ENEM2024_SP.xlsx",
    index=False
)

print("\n")
print("=" * 80)
print("ARQUIVO SALVO")
print("=" * 80)

print(
    "comparacao_HLM2_ML_ENEM2024_SP.xlsx"
)

df_modelo.to_csv('df_modelo.csv', index=False) 