import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#dados obtidos:
data = {
    'Experimento': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Temperatura_Real': [70, 70, 70, 70, 110, 110, 110, 110, 90, 90, 90],
    'Tempo_Real':       [10, 10, 30, 30, 10, 10, 30, 30, 20, 20, 20],
    'Concentracao_Real':[0.5, 2.5, 0.5, 2.5, 0.5, 2.5, 0.5, 2.5, 1.5, 1.5, 1.5],
    'T_cod': [-1, -1, -1, -1,  1,  1,  1,  1,  0,  0,  0],
    't_cod': [-1, -1,  1,  1, -1, -1,  1,  1,  0,  0,  0],
    'C_cod': [-1,  1, -1,  1, -1,  1, -1,  1,  0,  0,  0],
    'Acucar_Resultado': [5.2, 6.8, 4.5, 6.1, 8.1, 9.5, 7.5, 9.0, 8.5, 8.6, 8.4]
}
df = pd.DataFrame(data)

col_resposta = 'Acucar_Resultado'
unidade_resposta = 'g/L'

fator_temp_cod = 'T_cod'
fator_tempo_cod = 't_cod'
fator_conc_cod = 'C_cod'

mapa_nomes_reais_unidades = {
    fator_temp_cod:  ('Temperatura_Real', '°C'),
    fator_tempo_cod: ('Tempo_Real', 'min'),
    fator_conc_cod:  ('Concentracao_Real', '% v/v'),
}
NIVEL_FATOR_FIXO_COD_PADRAO = 0

# ---------------------------------------------------------------------------
# PASSO 3: AJUSTAR O MODELO DE REGRESSÃO DE SEGUNDA ORDEM (UMA VEZ)
# ---------------------------------------------------------------------------
formula_modelo = (
    f"{col_resposta} ~ "
    f"{fator_temp_cod} + {fator_tempo_cod} + {fator_conc_cod} + "
    f"I({fator_temp_cod}**2) + I({fator_tempo_cod}**2) + I({fator_conc_cod}**2) + "
    f"{fator_temp_cod}:{fator_tempo_cod} + "
    f"{fator_temp_cod}:{fator_conc_cod} + "
    f"{fator_tempo_cod}:{fator_conc_cod}"
)
modelo_global = smf.ols(formula=formula_modelo, data=df).fit()
print("Resumo do Modelo de Regressão Global:")
print(modelo_global.summary())
print("\n" + "="*80 + "\n")

def decodificar_valor_para_plot(val_cod, fator_cod_nome, df_original, mapa_nomes):
    fator_real_col = mapa_nomes[fator_cod_nome][0]
    val_min_real = df_original.loc[df_original[fator_cod_nome] == -1, fator_real_col].mean()
    val_ctr_real = df_original.loc[df_original[fator_cod_nome] == 0, fator_real_col].mean()
    val_max_real = df_original.loc[df_original[fator_cod_nome] == 1, fator_real_col].mean()

    if pd.isna(val_min_real) or pd.isna(val_ctr_real) or pd.isna(val_max_real):
        print(f"Aviso de decodificação: Faltam níveis (-1,0,1) para {fator_cod_nome}. Usando interpolação com min/max observados.")
        min_cod_obs, max_cod_obs = df_original[fator_cod_nome].min(), df_original[fator_cod_nome].max()
        min_real_obs = df_original.loc[df_original[fator_cod_nome] == min_cod_obs, fator_real_col].mean()
        max_real_obs = df_original.loc[df_original[fator_cod_nome] == max_cod_obs, fator_real_col].mean()
        if max_cod_obs == min_cod_obs: return min_real_obs
        return np.interp(val_cod, [min_cod_obs, max_cod_obs], [min_real_obs, max_real_obs])

    if val_cod == 0: return val_ctr_real
    if val_cod == -1: return val_min_real
    if val_cod == 1: return val_max_real
    if val_cod < 0: return np.interp(val_cod, [-1, 0], [val_min_real, val_ctr_real])
    else: return np.interp(val_cod, [0, 1], [val_ctr_real, val_max_real])

def plotar_superficie_e_contorno(modelo, df_dados_originais,
                                 eixo_X_fator_cod, eixo_Y_fator_cod, fator_fixo_cod,
                                 nivel_fator_fixo_cod_val,
                                 col_resp, unidade_resp, mapa_nomes, num_pontos_grade=30):
    print(f"Gerando gráficos para: X={eixo_X_fator_cod}, Y={eixo_Y_fator_cod}, Fixo={fator_fixo_cod} em {nivel_fator_fixo_cod_val}")

    x_vals_cod = np.linspace(df_dados_originais[eixo_X_fator_cod].min(), df_dados_originais[eixo_X_fator_cod].max(), num_pontos_grade)
    y_vals_cod = np.linspace(df_dados_originais[eixo_Y_fator_cod].min(), df_dados_originais[eixo_Y_fator_cod].max(), num_pontos_grade)
    X_grid_cod, Y_grid_cod = np.meshgrid(x_vals_cod, y_vals_cod)

    dados_predicao = pd.DataFrame({
        eixo_X_fator_cod: X_grid_cod.ravel(),
        eixo_Y_fator_cod: Y_grid_cod.ravel(),
        fator_fixo_cod: nivel_fator_fixo_cod_val
    })
    todos_fatores_principais_cod = [fator_temp_cod, fator_tempo_cod, fator_conc_cod]
    for fator_principal_cod in todos_fatores_principais_cod:
        if fator_principal_cod not in dados_predicao.columns:
            pass

    Z_predicao = modelo.predict(dados_predicao).values.reshape(X_grid_cod.shape)

    X_grid_real = np.array([[decodificar_valor_para_plot(val, eixo_X_fator_cod, df_dados_originais, mapa_nomes) for val in row] for row in X_grid_cod])
    Y_grid_real = np.array([[decodificar_valor_para_plot(val, eixo_Y_fator_cod, df_dados_originais, mapa_nomes) for val in row] for row in Y_grid_cod])
    val_fator_fixo_real = decodificar_valor_para_plot(nivel_fator_fixo_cod_val, fator_fixo_cod, df_dados_originais, mapa_nomes)

    nome_real_X, unidade_X = mapa_nomes[eixo_X_fator_cod]
    nome_real_Y, unidade_Y = mapa_nomes[eixo_Y_fator_cod]
    nome_real_fixo, unidade_fixo = mapa_nomes[fator_fixo_cod]

    label_eixo_x = f"{nome_real_X.replace('_Real','')} ({unidade_X})"
    label_eixo_y = f"{nome_real_Y.replace('_Real','')} ({unidade_Y})"
    label_fator_fixo_info = f"{nome_real_fixo.replace('_Real','')} = {val_fator_fixo_real:.2f} {unidade_fixo}"
    label_resposta_completo = f"{col_resp.replace('_Resultado','')} ({unidade_resp})"

    df_plot_pontos = df_dados_originais[df_dados_originais[fator_fixo_cod] == nivel_fator_fixo_cod_val]

    fig3d = plt.figure(figsize=(11, 7))
    ax3d = fig3d.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(X_grid_real, Y_grid_real, Z_predicao, cmap='viridis', edgecolor='none', alpha=0.85)
    if not df_plot_pontos.empty:
        ax3d.scatter(df_plot_pontos[nome_real_X], df_plot_pontos[nome_real_Y], df_plot_pontos[col_resp],
                     color='red', s=70, depthshade=True, label='Pontos Experimentais')
    ax3d.set_xlabel(label_eixo_x, fontsize=9); ax3d.set_ylabel(label_eixo_y, fontsize=9); ax3d.set_zlabel(label_resposta_completo, fontsize=9)
    ax3d.set_title(f"Superfície: {label_resposta_completo}\nvs {label_eixo_x} e {label_eixo_y}\n({label_fator_fixo_info})", fontsize=11)
    fig3d.colorbar(surf, shrink=0.5, aspect=10, label=label_resposta_completo)
    if not df_plot_pontos.empty: ax3d.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(9, 6))
    contour_fill = plt.contourf(X_grid_real, Y_grid_real, Z_predicao, 15, cmap='viridis', alpha=0.9)
    contour_lines = plt.contour(X_grid_real, Y_grid_real, Z_predicao, 15, colors='black', linewidths=0.7)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f')
    if not df_plot_pontos.empty:
        plt.scatter(df_plot_pontos[nome_real_X], df_plot_pontos[nome_real_Y], c=df_plot_pontos[col_resp],
                    cmap='viridis', edgecolor='black', s=90, label='Pontos Experimentais')
    plt.xlabel(label_eixo_x, fontsize=10); plt.ylabel(label_eixo_y, fontsize=10)
    plt.title(f"Contorno: {label_resposta_completo}\nvs {label_eixo_x} e {label_eixo_y}\n({label_fator_fixo_info})", fontsize=11)
    plt.colorbar(contour_fill, label=f"{label_resposta_completo} (Predita)")
    plt.grid(True, linestyle=':', alpha=0.6)
    if not df_plot_pontos.empty: plt.legend(loc='best'); plt.tight_layout(); plt.show()

plot_combinations = [
    {'x': fator_temp_cod, 'y': fator_tempo_cod, 'fix': fator_conc_cod},
    {'x': fator_temp_cod, 'y': fator_conc_cod, 'fix': fator_tempo_cod},
    {'x': fator_tempo_cod, 'y': fator_conc_cod, 'fix': fator_temp_cod}
]

for combo in plot_combinations:
    plotar_superficie_e_contorno(
        modelo=modelo_global, df_dados_originais=df,
        eixo_X_fator_cod=combo['x'],
        eixo_Y_fator_cod=combo['y'],
        fator_fixo_cod=combo['fix'],
        nivel_fator_fixo_cod_val=NIVEL_FATOR_FIXO_COD_PADRAO,
        col_resp=col_resposta, unidade_resp=unidade_resposta,
        mapa_nomes=mapa_nomes_reais_unidades
    )

print("\n" + "="*80)
print("Gerando Gráfico de Pareto dos Efeitos do Modelo Global...")

params = modelo_global.params.copy()
if 'Intercept' in params.index:
    effects_params = params.drop('Intercept')
else:
    print("Aviso: 'Intercept' não encontrado nos parâmetros. Usando todos para o Pareto.")
    effects_params = params

abs_effects = np.abs(effects_params)
sorted_effects = abs_effects.sort_values(ascending=False)
cumulative_percentage = (sorted_effects.cumsum() / sorted_effects.sum()) * 100

simplified_labels = []
for label in sorted_effects.index:
    new_label = label.replace('I(', '').replace(' ** 2)', '²').replace(')', '')
    new_label = new_label.replace(':', '*')
    new_label = new_label.replace(fator_temp_cod, 'T')
    new_label = new_label.replace(fator_tempo_cod, 't')
    new_label = new_label.replace(fator_conc_cod, 'C')
    simplified_labels.append(new_label)

fig_pareto, ax1_pareto = plt.subplots(figsize=(12, 7))
color_bar = 'royalblue'
bars = ax1_pareto.bar(simplified_labels, sorted_effects.values, color=color_bar, alpha=0.85, label='Magnitude do Efeito') # alpha aumentado
ax1_pareto.set_xlabel('Termos do Modelo (Fatores e Interações)', fontsize=12)
ax1_pareto.set_ylabel('Magnitude Absoluta do Coeficiente (Efeito)', color=color_bar, fontsize=12)
ax1_pareto.tick_params(axis='y', labelcolor=color_bar)
ax1_pareto.tick_params(axis='x', rotation=45, ha="right", fontsize=10)

for bar in bars:
    yval = bar.get_height()
    ax1_pareto.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015 * sorted_effects.max(), f'{yval:.3f}',
                    ha='center', va='bottom', fontsize=9, color='black') #cor txt

ax2_pareto = ax1_pareto.twinx()
color_line = 'red'
ax2_pareto.plot(simplified_labels, cumulative_percentage.values, color=color_line, marker='o', linestyle='-',
                linewidth=2.5, markersize=6, label='Porcentagem Cumulativa')
ax2_pareto.set_ylabel('Porcentagem Cumulativa (%)', color=color_line, fontsize=12)
ax2_pareto.tick_params(axis='y', labelcolor=color_line)
ax2_pareto.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2_pareto.set_ylim(0, 105)

fig_pareto.suptitle(f'Gráfico de Pareto dos Efeitos para {col_resposta.replace("_Resultado","")} ({unidade_resposta})',
                    fontsize=14, fontweight='bold')
ax1_pareto.grid(True, axis='y', linestyle=':', alpha=0.6, which='major')
fig_pareto.tight_layout(rect=[0, 0.03, 1, 0.95])
lines, labels_ax1 = ax1_pareto.get_legend_handles_labels()
lines2, labels_ax2 = ax2_pareto.get_legend_handles_labels()
ax2_pareto.legend(lines + lines2, labels_ax1 + labels_ax2, loc='center right', fontsize=10) #fontsize legenda

plt.show()
print("Gráfico de Pareto gerado.")
print("Script finalizado.")
