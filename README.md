Modelo de Regressão para Previsão de Preços de Imóveis

Este projeto detalha a construção de um modelo de machine learning para prever o preço de venda de imóveis. Utilizando um conjunto de dados com características como tipo de imóvel, bairro, área, quartos e diferenciais (amenidades), o objetivo é criar o modelo de regressão mais preciso possível.

Metodologia

O desenvolvimento foi focado em extrair o máximo de informação das features disponíveis, seguindo o pipeline abaixo:

1. Carga e Limpeza de Dados

    Os dados de treino e teste foram carregados e a coluna Id, por ser um identificador único sem valor preditivo, foi removida.

    (Mencionar qualquer outro tratamento de valores ausentes ou limpeza inicial).

2. Engenharia de Features (Etapa Crítica)

Esta foi a etapa mais importante do projeto. As features categóricas de alta cardinalidade (diferenciais e bairro) foram tratadas para se tornarem úteis ao modelo:

    Subcategorização de Diferenciais: A coluna diferenciais continha mais de 50 combinações únicas de amenidades (ex: 'campo de futebol e copa', 'churrasqueira e piscina'). Elas foram processadas e agrupadas em categorias-raiz (ex: 'futebol+', 'churrasco+', 'piscina+'), transformando uma coluna de texto complexa em features binárias/numéricas de alta qualidade.

    Agrupamento de Bairros: Para evitar a alta cardinalidade da feature bairro, os bairros foram agrupados (ex: por perfil socioeconômico ou zona, conforme descrito no relatório) para reduzir a dimensionalidade e ajudar o modelo a generalizar.

    Tratamento de Outliers: Foi realizada uma análise para identificar e tratar outliers de preço que poderiam distorcer o treinamento dos modelos.

3. Seleção de Features e Pré-processamento

    Seleção de Features: A Correlação de Pearson foi utilizada para analisar a relação linear entre as features numéricas e o preço-alvo, ajudando a selecionar as variáveis mais relevantes.

    Normalização: O StandardScaler do Scikit-learn foi aplicado às features numéricas. Isso é essencial para o bom desempenho de modelos lineares e do KNN, que são sensíveis à escala.

4. Modelagem e Comparação

Quatro algoritmos de regressão diferentes foram treinados e avaliados:

    Regressão Linear: Usado como baseline para estabelecer um desempenho mínimo.

    Regressão Polinomial (Graus 2 e 3): Para capturar relações não-lineares.

    K-Neighbors Regressor (KNN): Um modelo não-paramétrico baseado em distância.

    Random Forest Regressor: Um modelo de ensemble robusto, excelente para capturar interações complexas sem sofrer overfitting facilmente.

5. Validação

    Validação Cruzada (K-Fold): Foi utilizada a validação cruzada KFold (com 5 splits) para avaliar a performance de generalização dos modelos.

    Métrica de Avaliação: O R2 (Coeficiente de Determinação) foi a métrica primária para medir a proporção da variância no preço que é explicável pelo modelo.

Resultados

A comparação dos modelos revelou insights claros:

    Regressão Linear: Atingiu um R2 de ~65%, servindo como um baseline razoável.

    Regressão Polinomial: Apresentou overfitting severo. Embora o R2 no treino fosse alto, o RMSE no teste explodiu, indicando que o modelo estava memorizando os dados de treino e não generalizava.

    Random Forest Regressor: Foi o modelo de melhor desempenho, alcançando um R2 médio de ~81% na validação cruzada. Isso demonstra sua capacidade superior de lidar com relações não-lineares complexas e a alta dimensionalidade das features criadas.

Tecnologias Utilizadas

    Python

    Pandas: Para manipulação de dados e engenharia de features.

    NumPy: Para operações numéricas.

    SciPy: Para análise estatística (Correlação de Pearson).

    Scikit-learn:

        Pré-processamento: StandardScaler

        Modelagem: LinearRegression, PolynomialFeatures, KNeighborsRegressor, RandomForestRegressor

        Validação: KFold, cross_val_score, r2_score

  Relatório Técnico Completo

Para uma análise detalhada da engenharia de features, tratamento de outliers, e a discussão aprofundada sobre a comparação dos modelos (incluindo o overfitting da Regressão Polinomial), consulte o relatório técnico completo:
