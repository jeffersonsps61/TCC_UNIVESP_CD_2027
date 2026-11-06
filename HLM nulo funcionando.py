# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:49:04 2026

@author: jefferson.santos
"""
# -*- coding: utf-8 -*-

# =============================================================================
# %% HLM2 NULO - ESTIMAÇÃO NUMERICAMENTE ESTÁVEL
# =============================================================================
#
# Base utilizada:
#     df_modelo
#
# Estrutura:
#     Nível 1 = estudante
#     Nível 2 = escola
#
# Modelo:
#
#     Y_ij = γ00 + u_0j + e_ij
#
# Onde:
#     Y_ij  = desempenho médio do estudante i na escola j
#     γ00   = intercepto / média geral
#     u_0j  = efeito aleatório da escola
#     e_ij  = erro individual
#
# Objetivo:
#     Estimar a variância entre escolas e calcular o ICC.
#
# =============================================================================


# =============================================================================
# %% 1. BIBLIOTECAS
# =============================================================================

import warnings
import numpy as np
import pandas as pd

import statsmodels.formula.api as smf


# =============================================================================
# %% 2. VERIFICAR DF_MODELO
# =============================================================================


df_modelo = pd.read_csv('df_modelo.csv')
     
     
print("=" * 80)
print("VERIFICAÇÃO DO DF_MODELO")
print("=" * 80)

if "df_modelo" not in globals():

    raise NameError(
        "A variável 'df_modelo' não está carregada."
    )


print("\nDimensão do df_modelo:")
print(df_modelo.shape)


print("\nNúmero de estudantes:")
print(f"{len(df_modelo):,}")


print("\nNúmero de escolas:")
print(
    df_modelo["CO_ESCOLA"].nunique()
)


# =============================================================================
# %% 3. VERIFICAR VARIÁVEIS
# =============================================================================

print("\n" + "=" * 80)
print("VARIÁVEIS UTILIZADAS")
print("=" * 80)


VAR_Y = "desempenho_medio"
VAR_GRUPO = "CO_ESCOLA"


if VAR_Y not in df_modelo.columns:

    raise KeyError(
        f"A variável '{VAR_Y}' não existe no df_modelo."
    )


if VAR_GRUPO not in df_modelo.columns:

    raise KeyError(
        f"A variável '{VAR_GRUPO}' não existe no df_modelo."
    )


print(f"Variável dependente : {VAR_Y}")
print(f"Grupo                : {VAR_GRUPO}")


# =============================================================================
# %% 4. CRIAR CÓPIA SOMENTE PARA O HLM2
# =============================================================================

df_hlm2 = df_modelo[
    [
        VAR_Y,
        VAR_GRUPO
    ]
].copy()


# =============================================================================
# %% 5. LIMPEZA MÍNIMA
# =============================================================================

print("\n" + "=" * 80)
print("LIMPEZA DA BASE")
print("=" * 80)


# Converter desempenho para numérico

df_hlm2[VAR_Y] = pd.to_numeric(
    df_hlm2[VAR_Y],
    errors="coerce"
)


# Converter escola para string

df_hlm2[VAR_GRUPO] = (
    df_hlm2[VAR_GRUPO]
    .astype("string")
)


# Remover missing

n_antes = len(df_hlm2)


df_hlm2 = df_hlm2.dropna(
    subset=[
        VAR_Y,
        VAR_GRUPO
    ]
).copy()


n_depois = len(df_hlm2)


print(
    f"Observações antes : {n_antes:,}"
)

print(
    f"Observações depois: {n_depois:,}"
)

print(
    f"Removidas         : {n_antes - n_depois:,}"
)


# =============================================================================
# %% 6. ESTRUTURA FINAL
# =============================================================================

N = len(df_hlm2)

J = df_hlm2[VAR_GRUPO].nunique()


print("\n" + "=" * 80)
print("ESTRUTURA DO HLM2")
print("=" * 80)


print(
    f"""
Estudantes (N): {N:,}
Escolas (J)   : {J:,}
"""


)


# =============================================================================
# %% 7. TAMANHO DOS GRUPOS
# =============================================================================

tamanho_escolas = (
    df_hlm2
    .groupby(VAR_GRUPO)
    .size()
)


print("\nDistribuição do número de estudantes por escola:")

print(
    tamanho_escolas.describe()
)


print(
    f"\nMínimo: {tamanho_escolas.min()}"
)

print(
    f"Máximo: {tamanho_escolas.max()}"
)


# =============================================================================
# %% 8. MÉDIA EMPÍRICA DA NOTA
# =============================================================================

media_y = df_hlm2[VAR_Y].mean()

desvio_y = df_hlm2[VAR_Y].std()

variancia_y = df_hlm2[VAR_Y].var()


print("\n" + "=" * 80)
print("ESTATÍSTICAS DA VARIÁVEL DEPENDENTE")
print("=" * 80)


print(
    f"""
Média:
    {media_y:.6f}

Desvio-padrão:
    {desvio_y:.6f}

Variância:
    {variancia_y:.6f}
"""
)


# =============================================================================
# %% 9. MODELO HLM2 NULO
# =============================================================================

print("\n" + "=" * 80)
print("ESPECIFICAÇÃO DO MODELO")
print("=" * 80)


formula_nula = f"{VAR_Y} ~ 1"


print(
    f"""
Modelo:

    {VAR_Y}_ij = γ00 + u_0j + e_ij

Fórmula statsmodels:

    {formula_nula}

Efeito aleatório:

    intercepto por escola
"""
)


# =============================================================================
# %% 10. FUNÇÃO DE ESTIMAÇÃO
# =============================================================================

def ajustar_hlm2_nulo(
    data,
    formula,
    grupo,
    metodo,
    maxiter=3000
):

    """
    Estima o HLM2 nulo por máxima verossimilhança (ML).

    Retorna:
        resultado
        warnings
        erro
    """

    resultado = None

    avisos = []

    erro = None


    try:

        # -------------------------------------------------------------
        # Criar modelo
        # -------------------------------------------------------------

        modelo = smf.mixedlm(
            formula=formula,
            data=data,
            groups=data[grupo],
            re_formula="1"
        )


        # -------------------------------------------------------------
        # Estimação
        # -------------------------------------------------------------

        with warnings.catch_warnings(record=True) as w:

            warnings.simplefilter("always")


            resultado = modelo.fit(
                reml=False,
                method=metodo,
                maxiter=maxiter,
                disp=False
            )


            avisos = [
                str(x.message)
                for x in w
            ]


    except Exception as e:

        erro = str(e)


    return resultado, avisos, erro


# =============================================================================
# %% 11. OTIMIZADORES
# =============================================================================

otimizadores = [
    "lbfgs",
    "bfgs",
    "cg",
    "powell"
]


print("\n" + "=" * 80)
print("OTIMIZADORES A SEREM TESTADOS")
print("=" * 80)


for metodo in otimizadores:

    print(
        f" - {metodo}"
    )


# =============================================================================
# %% 12. ESTIMAR HLM2 COM CADA OTIMIZADOR
# =============================================================================

resultados = {}

tabela_resultados = []


print("\n" + "=" * 80)
print("INICIANDO ESTIMAÇÕES")
print("=" * 80)


for metodo in otimizadores:

    print("\n" + "-" * 80)

    print(
        f"OTIMIZADOR: {metodo.upper()}"
    )

    print("-" * 80)


    resultado, avisos, erro = ajustar_hlm2_nulo(
        data=df_hlm2,
        formula=formula_nula,
        grupo=VAR_GRUPO,
        metodo=metodo,
        maxiter=3000
    )


    # -----------------------------------------------------------------
    # Se houve erro
    # -----------------------------------------------------------------

    if resultado is None:

        print("❌ Falha na estimação.")

        print(
            f"Erro: {erro}"
        )


        tabela_resultados.append({

            "otimizador": metodo,
            "convergiu": False,
            "loglik": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "var_escola": np.nan,
            "var_residual": np.nan,
            "ICC": np.nan,
            "n_warnings": len(avisos),
            "loglik_finito": False,
            "variancias_validas": False,
            "solucao_valida": False,
            "erro": erro

        })

        continue


    # -----------------------------------------------------------------
    # Extrair LogLik
    # -----------------------------------------------------------------

    try:

        loglik = float(resultado.llf)

    except:

        loglik = np.nan


    # -----------------------------------------------------------------
    # AIC
    # -----------------------------------------------------------------

    try:

        aic = float(resultado.aic)

    except:

        aic = np.nan


    # -----------------------------------------------------------------
    # BIC
    # -----------------------------------------------------------------

    try:

        bic = float(resultado.bic)

    except:

        bic = np.nan


    # -----------------------------------------------------------------
    # Variância entre escolas
    # -----------------------------------------------------------------

    try:

        var_escola = float(
            resultado.cov_re.iloc[0, 0]
        )

    except:

        var_escola = np.nan


    # -----------------------------------------------------------------
    # Variância residual
    # -----------------------------------------------------------------

    try:

        var_residual = float(
            resultado.scale
        )

    except:

        var_residual = np.nan


    # -----------------------------------------------------------------
    # ICC
    # -----------------------------------------------------------------

    if (
        np.isfinite(var_escola)
        and
        np.isfinite(var_residual)
        and
        var_escola >= 0
        and
        var_residual > 0
    ):

        icc = (
            var_escola /
            (
                var_escola +
                var_residual
            )
        )

    else:

        icc = np.nan


    # -----------------------------------------------------------------
    # Convergência
    # -----------------------------------------------------------------

    convergiu = bool(
        getattr(
            resultado,
            "converged",
            False
        )
    )


    # -----------------------------------------------------------------
    # LogLik finito?
    # -----------------------------------------------------------------

    loglik_finito = bool(
        np.isfinite(loglik)
    )


    # -----------------------------------------------------------------
    # Variâncias válidas?
    # -----------------------------------------------------------------

    variancias_validas = bool(
        np.isfinite(var_escola)
        and
        np.isfinite(var_residual)
        and
        var_escola >= 0
        and
        var_residual > 0
    )


    # -----------------------------------------------------------------
    # Solução numericamente válida
    # -----------------------------------------------------------------

    solucao_valida = bool(
        convergiu
        and
        loglik_finito
        and
        variancias_validas
    )


    # -----------------------------------------------------------------
    # Guardar
    # -----------------------------------------------------------------

    resultados[metodo] = resultado


    tabela_resultados.append({

        "otimizador": metodo,

        "convergiu": convergiu,

        "loglik": loglik,

        "AIC": aic,

        "BIC": bic,

        "var_escola": var_escola,

        "var_residual": var_residual,

        "ICC": icc,

        "n_warnings": len(avisos),

        "loglik_finito": loglik_finito,

        "variancias_validas": variancias_validas,

        "solucao_valida": solucao_valida,

        "erro": erro

    })


    # -----------------------------------------------------------------
    # Mostrar
    # -----------------------------------------------------------------

    print(
        f"Convergiu       : {convergiu}"
    )

    print(
        f"LogLik          : {loglik}"
    )

    print(
        f"Variância escola: {var_escola}"
    )

    print(
        f"Variância erro  : {var_residual}"
    )

    print(
        f"ICC             : {icc}"
    )

    print(
        f"Warnings        : {len(avisos)}"
    )

    print(
        f"Solução válida  : {solucao_valida}"
    )


    # -----------------------------------------------------------------
    # Mostrar warnings
    # -----------------------------------------------------------------

    if len(avisos) > 0:

        print("\nWarnings encontrados:")

        for aviso in avisos:

            print(
                "  -",
                aviso
            )


# =============================================================================
# %% 13. TABELA DE COMPARAÇÃO
# =============================================================================

df_comparacao_otimizadores = pd.DataFrame(
    tabela_resultados
)


print("\n" + "=" * 80)
print("COMPARAÇÃO DOS OTIMIZADORES")
print("=" * 80)


print(
    df_comparacao_otimizadores.to_string(
        index=False
    )
)


# =============================================================================
# %% 14. IDENTIFICAR SOLUÇÕES VÁLIDAS
# =============================================================================

validos = (
    df_comparacao_otimizadores[
        df_comparacao_otimizadores[
            "solucao_valida"
        ]
        == True
    ]
    .copy()
)


print("\n" + "=" * 80)
print("SOLUÇÕES VÁLIDAS")
print("=" * 80)


print(
    f"\nNúmero de soluções válidas: {len(validos)}"
)


if len(validos) == 0:

    raise RuntimeError(
        """
        Nenhum otimizador produziu uma solução
        numericamente válida.

        NÃO devemos calcular ou interpretar o ICC.

        Será necessário investigar a parametrização
        e a estrutura da base.
        """
    )


# =============================================================================
# %% 15. ESCOLHER A SOLUÇÃO
# =============================================================================
#
# Entre as soluções que:
#
#     - convergiram;
#     - possuem LogLik finito;
#     - possuem variâncias válidas;
#
# selecionamos aquela com maior LogLik.
#
# =============================================================================

melhor = (
    validos
    .sort_values(
        "loglik",
        ascending=False
    )
    .iloc[0]
)


melhor_metodo = melhor["otimizador"]


hlm2_nulo = resultados[
    melhor_metodo
]


print("\n" + "=" * 80)
print("MODELO SELECIONADO")
print("=" * 80)


print(
    f"""
Otimizador:
    {melhor_metodo}

LogLik:
    {hlm2_nulo.llf:.6f}

Convergiu:
    {hlm2_nulo.converged}
"""
)


# =============================================================================
# %% 16. SUMMARY COMPLETO
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY DO HLM2 NULO")
print("=" * 80)


print(
    hlm2_nulo.summary()
)


# =============================================================================
# %% 17. COMPONENTES DE VARIÂNCIA
# =============================================================================

tau_00 = float(
    hlm2_nulo.cov_re.iloc[0, 0]
)


sigma_2 = float(
    hlm2_nulo.scale
)


print("\n" + "=" * 80)
print("COMPONENTES DE VARIÂNCIA")
print("=" * 80)


print(
    f"""
Variância entre escolas (τ00):
    {tau_00:.10f}

Variância residual (σ²):
    {sigma_2:.10f}
"""
)


# =============================================================================
# %% 18. VARIÂNCIA TOTAL
# =============================================================================

variancia_total = (
    tau_00 +
    sigma_2
)


print(
    f"""
Variância total:

τ00 + σ² =
{variancia_total:.10f}
"""
)


# =============================================================================
# %% 19. ICC
# =============================================================================

print("\n" + "=" * 80)
print("ICC - COEFICIENTE DE CORRELAÇÃO INTRACLASSE")
print("=" * 80)


if (
    np.isfinite(tau_00)
    and
    np.isfinite(sigma_2)
    and
    tau_00 >= 0
    and
    sigma_2 > 0
    and
    variancia_total > 0
):

    ICC = (
        tau_00 /
        variancia_total
    )


    print(
        f"""
Fórmula:

ICC = τ00 / (τ00 + σ²)


ICC = {tau_00:.10f}
      ------------------------
      {tau_00:.10f} + {sigma_2:.10f}


ICC = {ICC:.10f}


ICC (%) = {ICC * 100:.4f}%
"""
    )

else:

    ICC = np.nan

    print(
        """
        ICC não pôde ser calculado porque
        os componentes de variância não são
        numericamente válidos.
        """
    )


# =============================================================================
# %% 20. DIAGNÓSTICO DA VARIÂNCIA ESCOLAR
# =============================================================================

print("\n" + "=" * 80)
print("DIAGNÓSTICO DA VARIÂNCIA ENTRE ESCOLAS")
print("=" * 80)


TOL = 1e-8


if tau_00 <= TOL:

    print(
        """
        A variância entre escolas está essencialmente
        no limite inferior da região admissível.

        Isso caracteriza uma possível solução de fronteira.

        O resultado deve ser interpretado com cautela.
        """
    )

else:

    print(
        f"""
        A variância entre escolas é positiva:

        τ00 = {tau_00:.10f}

        Portanto, o modelo estima heterogeneidade
        entre as escolas.
        """
    )


# =============================================================================
# %% 21. DIAGNÓSTICO DO LOGLIK
# =============================================================================

print("\n" + "=" * 80)
print("DIAGNÓSTICO DA LOG-VEROSSIMILHANÇA")
print("=" * 80)


if np.isfinite(hlm2_nulo.llf):

    print(
        f"LogLik = {hlm2_nulo.llf:.6f}"
    )

    print(
        "\nLogLik FINITO."
    )

    print(
        "O modelo pode ser utilizado em comparações "
        "posteriores por ML."
    )

else:

    print(
        "\nLogLik NÃO FINITO."
    )

    print(
        "Não utilizar este modelo para teste LR/AIC/BIC."
    )


# =============================================================================
# %% 22. DIAGNÓSTICO DA CONVERGÊNCIA
# =============================================================================

print("\n" + "=" * 80)
print("DIAGNÓSTICO DA CONVERGÊNCIA")
print("=" * 80)


print(
    f"Convergiu = {hlm2_nulo.converged}"
)


# =============================================================================
# %% 23. COMPARAR MÉDIA EMPÍRICA E INTERCEPTO
# =============================================================================

intercepto = float(
    hlm2_nulo.fe_params.iloc[0]
)


print("\n" + "=" * 80)
print("INTERCEPTO")
print("=" * 80)


print(
    f"""
Média empírica:
    {media_y:.6f}

Intercepto estimado:
    {intercepto:.6f}

Diferença:
    {intercepto - media_y:.6f}
"""
)


# =============================================================================
# %% 24. RESUMO FINAL
# =============================================================================

print("\n" + "=" * 80)
print("RESUMO FINAL DO HLM2 NULO")
print("=" * 80)


print(
    f"""
N estudantes:
    {N:,}

N escolas:
    {J:,}

Otimizador:
    {melhor_metodo}

Convergiu:
    {hlm2_nulo.converged}

LogLik:
    {hlm2_nulo.llf:.6f}

Variância entre escolas (τ00):
    {tau_00:.6f}

Variância residual (σ²):
    {sigma_2:.6f}

Variância total:
    {variancia_total:.6f}

ICC:
    {ICC:.6f}

ICC (%):
    {ICC * 100:.4f}%
"""
)


# =============================================================================
# %% 25. TABELA FINAL PARA O TCC
# =============================================================================

resultado_hlm2_nulo = pd.DataFrame({

    "Modelo": [
        "HLM2 Nulo"
    ],

    "N_estudantes": [
        N
    ],

    "N_escolas": [
        J
    ],

    "Otimizador": [
        melhor_metodo
    ],

    "Convergiu": [
        hlm2_nulo.converged
    ],

    "LogLik": [
        hlm2_nulo.llf
    ],

    "AIC": [
        hlm2_nulo.aic
    ],

    "BIC": [
        hlm2_nulo.bic
    ],

    "Tau_00_variancia_escola": [
        tau_00
    ],

    "Sigma_2_variancia_residual": [
        sigma_2
    ],

    "Variancia_total": [
        variancia_total
    ],

    "ICC": [
        ICC
    ],

    "ICC_percentual": [
        ICC * 100
    ]

})


print("\n" + "=" * 80)
print("TABELA FINAL")
print("=" * 80)


print(
    resultado_hlm2_nulo.to_string(
        index=False
    )
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