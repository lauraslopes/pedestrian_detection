Relatório trabalho 4 Visão Computacional
Aluna: Laura Silva Lopes, GRR20163048

Os caminhos (linha 148 e linha 150) até os diretórios com as imagens de treino estão como:
path_pos = 'INRIAPerson/96X160H96/Train/pos'
path_neg = 'INRIAPerson/Train/neg'

Estão implementados e executando corretamente no código:
- Leitura do banco de imagens
- Cálculo do HOG
- Treino SVM
- Hardmining
- Sliding Window com pirâmides de imagens

Após o cálculo dos HOGs, as features de cada imagem são salvas em um arquivo .npy, portanto, para uma
alteração no código que não envolva a função de HOG, é possível apenas comentar a chamada da função e usar os resultados
previamente calculados e salvos nos arquivos negoutput.npy, para as features das imagens negativas, e posoutput, para as
features das imagens positivas.
Os processos de Hardmining e Sliding Window são extremamente demorados, por esse motivo não foi possível obter um resultado.

Bugs encontrados: mesmo com o uso de Numba, biblioteca de otimização, na obtenção dos HOGs das imagens, a função ainda é lenta.
