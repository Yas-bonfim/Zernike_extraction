
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes


# CONFIGURACOES

# Imagem 2D da fase.
# Ajuste para o caminho da imagem que deseja analisar.
IMAGE_PATH = "Z4_amp-0.01.png"

# Numero de coeficientes de Zernike a estimar.
# 15 modos: ate ordem radial n=4.
# 36 modos: ate ordem radial n=7.
N_COEFFICIENTS = 15

# Limites da fase representada pelas cores.
# IMPORTANTE: confirmar na barra de cores da figura original.
PHASE_MIN = -0.025
PHASE_MAX = 0.025

# Mapa de cores utilizado na figura.
CMAP_NAME = "turbo"

# Centro e raio da pupila em pixels.
# None: tenta detectar automaticamente.
# Se a deteccao falhar, informe manualmente.
PUPIL_CENTER = None  # (cx, cy)
PUPIL_RADIUS = None  # pixels

# Coordenadas fisicas da pupila, se conhecidas
# Exemplo: raio fisico de 6.3 mm
PUPIL_RADIUS_MM = 6.3


# CARREGAR A IMAGEM

img = Image.open(IMAGE_PATH).convert("RGB")
rgb = np.asarray(img).astype(float) / 255.0

height, width, _ = rgb.shape

print("Dimensoes:", width, "x", height)


# DETECCAO DA PUPILA

# Detecta pixels que nao pertencem ao fundo preto
# Essa abordagem funciona melhor em imagens com fundo preto
brightness = np.max(rgb, axis=2)

mask_initial = brightness > 0.08

# Preenche eventuais pequenos buracos.
mask_initial = binary_fill_holes(mask_initial)

yy, xx = np.where(mask_initial)

if PUPIL_CENTER is None or PUPIL_RADIUS is None:

    if len(xx) == 0:
        raise ValueError(
            "Nao foi possivel detectar a pupila. "
            "Defina PUPIL_CENTER e PUPIL_RADIUS manualmente."
        )

    cx = np.mean(xx)
    cy = np.mean(yy)

    distances = np.sqrt(
        (xx - cx)**2 + (yy - cy)**2
    )

    radius = np.percentile(distances, 99.5)

else:

    cx, cy = PUPIL_CENTER
    radius = PUPIL_RADIUS


print(f"Centro da pupila: ({cx:.2f}, {cy:.2f})")
print(f"Raio da pupila: {radius:.2f} pixels")


# Coordenadas da imagem.
Y, X = np.indices((height, width))

rho = np.sqrt(
    (X - cx)**2 + (Y - cy)**2
) / radius

theta = np.arctan2(
    Y - cy,
    X - cx
)

pupil_mask = rho <= 1.0


# RECONSTRUCAO DA FASE A PARTIR DAS CORES

# Gera uma LUT com o mapa de cores utilizado
cmap = plt.get_cmap(CMAP_NAME)

lut_size = 4096

lut = cmap(
    np.linspace(0, 1, lut_size)
)[:, :3]

# Cada cor corresponde a um valor de fase
phase_lut = np.linspace(
    PHASE_MIN,
    PHASE_MAX,
    lut_size
)

# Busca a cor mais proxima para cada pixel
tree = cKDTree(lut)

pixels = rgb[pupil_mask]

_, indices = tree.query(pixels)

phase = np.full(
    (height, width),
    np.nan
)

phase[pupil_mask] = phase_lut[indices]


# POLINOMIOS DE ZERNIKE

def radial_polynomial(n, m, rho):

    m = abs(m)

    if (n - m) % 2 != 0:
        return np.zeros_like(rho)

    result = np.zeros_like(rho, dtype=float)

    for k in range((n - m) // 2 + 1):

        coefficient = (
            (-1)**k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
        )

        result += coefficient * rho**(n - 2*k)

    return result

    return result


def zernike_osa(n, m, rho, theta):

    R = radial_polynomial(n, m, rho)

    if m == 0:

        normalization = np.sqrt(n + 1)

        Z = normalization * R

    elif m > 0:

        normalization = np.sqrt(2 * (n + 1))

        Z = (
            normalization
            * R
            * np.cos(m * theta)
        )

    else:

        normalization = np.sqrt(2 * (n + 1))

        Z = (
            normalization
            * R
            * np.sin(abs(m) * theta)
        )

    return Z


# GERAR A BASE DE ZERNIKE

modes = []

n = 0

while len(modes) < N_COEFFICIENTS:

    for m in range(-n, n + 1, 2):

        modes.append((n, m))

        if len(modes) == N_COEFFICIENTS:
            break

    n += 1


# Coordenadas dentro da pupila.
rho_valid = rho[pupil_mask]
theta_valid = theta[pupil_mask]

phase_valid = phase[pupil_mask]



# AJUSTE POR MINIMOS QUADRADOS

Z_matrix = np.column_stack([
    zernike_osa(
        n,
        m,
        rho_valid,
        theta_valid
    )
    for n, m in modes
])

# Elimina pixels com valores invalidos
valid = np.isfinite(phase_valid)

Z_matrix_valid = Z_matrix[valid]

phase_valid = phase_valid[valid]

# Ajuste linear.
coefficients, residuals, rank, singular_values = (
    np.linalg.lstsq(
        Z_matrix_valid,
        phase_valid,
        rcond=None
    )
)

print("\nCoeficientes de Zernike:")
print("Rank da matriz:", rank)



# TABELA DE COEFICIENTES

df = pd.DataFrame({
    "OSA_index": np.arange(N_COEFFICIENTS),
    "n": [mode[0] for mode in modes],
    "m": [mode[1] for mode in modes],
    "Coefficient_rad": coefficients
})

print(df)

df.to_csv(
    "zernike_coefficients.csv",
    index=False
)



# RECONSTRUCAO DA FASE

phase_reconstructed = np.full(
    (height, width),
    np.nan
)

phase_reconstructed[pupil_mask] = (
    Z_matrix @ coefficients
)


# CALCULO DO ERRO RESIDUAL

residual = (
    phase[pupil_mask]
    - phase_reconstructed[pupil_mask]
)

# Remove valores infinitos/NaN
residual = residual[np.isfinite(residual)]

# Cálculo do RMS 
rms = np.sqrt(np.mean(residual**2))
print(f"\nRMS residual: {rms:.8f} rad")

# Cálculo da Variância Geral do Erro de Fase
# A variância mede a dispersão dos erros em torno da média
variancia_residual = np.var(residual)
print(f"Variância do erro residual: {variancia_residual:.8e} rad²")

# Conversão para Ruído de Excesso
# Em sistemas ópticos, a variância da fase residual representa diretamente 
# a potência do ruído de excesso (excess noise power) não corrigido.
ruido_excesso = variancia_residual 
print(f"Ruído de excesso (potência): {ruido_excesso:.8e} rad²")

# Se precisar do ruído de excesso em decibéis (dB) em relação a 1 rad²
if ruido_excesso > 0:
    ruido_excesso_db = 10 * np.log10(ruido_excesso)
    print(f"Ruído de excesso em dB: {ruido_excesso_db:.2f} dB")



# GRAFICOS


fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)

im0 = axes[0].imshow(
    phase,
    cmap=CMAP_NAME,
    vmin=PHASE_MIN,
    vmax=PHASE_MAX
)

axes[0].set_title("Fase extraida da imagem")

plt.colorbar(
    im0,
    ax=axes[0],
    label="Fase (rad)"
)


im1 = axes[1].imshow(
    phase_reconstructed,
    cmap=CMAP_NAME,
    vmin=PHASE_MIN,
    vmax=PHASE_MAX
)

axes[1].set_title("Fase reconstruida")

plt.colorbar(
    im1,
    ax=axes[1],
    label="Fase (rad)"
)


im2 = axes[2].imshow(
    phase - phase_reconstructed,
    cmap="coolwarm"
)

axes[2].set_title("Erro residual")

plt.colorbar(
    im2,
    ax=axes[2],
    label="Erro (rad)"
)


for ax in axes:
    ax.set_axis_off()

plt.tight_layout()
plt.show()



# GRAFICO DOS COEFICIENTES

# plt.figure(figsize=(12, 5))

# plt.bar(
#     df["OSA_index"],
#     df["Coefficient_rad"]
# )

# plt.xlabel("Indice OSA")
# plt.ylabel("Coeficiente (rad)")
# plt.title("Coeficientes de Zernike estimados")

# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()


# print("\nArquivo salvo: zernike_coefficients.csv")
