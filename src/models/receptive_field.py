# src/models/receptive_field.py

def compute_receptive_field(layers):
    """
    Calcula o campo receptivo teórico acumulado ao longo de uma
    sequência de camadas.

    layers: lista de dicts {"kernel": k, "stride": s}, na ordem em
            que os dados passam por elas (da entrada para a saída).

    Retorna: campo receptivo final (em pixels da imagem de entrada).
    """
    rf = 1       # campo receptivo acumulado
    jump = 1     # produto dos strides até aqui ("distância" entre posições consecutivas)

    for layer in layers:
        k, s = layer["kernel"], layer["stride"]
        rf = rf + (k - 1) * jump
        jump = jump * s

    return rf


def resunet_encoder_layers():
    """
    Sequência de camadas do ENCODER do nosso ResUNet (é o encoder que
    determina o campo receptivo no ponto mais profundo/abstrato --
    o bottleneck, que é o que "vê" mais contexto).

    Cada ResidualBlock tem 2 convs 3x3 stride 1; entre blocos, um
    MaxPool2d(2) (kernel 2, stride 2).
    """
    layers = []
    # enc1: 2 convs 3x3 stride 1
    layers += [{"kernel": 3, "stride": 1}, {"kernel": 3, "stride": 1}]
    layers += [{"kernel": 2, "stride": 2}]  # pool

    # enc2
    layers += [{"kernel": 3, "stride": 1}, {"kernel": 3, "stride": 1}]
    layers += [{"kernel": 2, "stride": 2}]  # pool

    # enc3
    layers += [{"kernel": 3, "stride": 1}, {"kernel": 3, "stride": 1}]
    layers += [{"kernel": 2, "stride": 2}]  # pool

    # enc4
    layers += [{"kernel": 3, "stride": 1}, {"kernel": 3, "stride": 1}]
    layers += [{"kernel": 2, "stride": 2}]  # pool

    # bottleneck
    layers += [{"kernel": 3, "stride": 1}, {"kernel": 3, "stride": 1}]

    return layers


if __name__ == "__main__":
    layers = resunet_encoder_layers()
    rf = compute_receptive_field(layers)
    print(f"Campo receptivo teórico (até o bottleneck): {rf}px")