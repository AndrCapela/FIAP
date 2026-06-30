from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
  
# Dados de exemplo
textos = [
      "O novo lançamento da Apple",
      "Resultado do jogo de ontem",
      "Eleições presidenciais",
      "Atualização no mundo da tecnologia",
      "Campeonato de futebol",
      "Política internacional",
      "Os Estados Unidos declarou guerra ao Irã",
      "O Brasil tem um novo presidente",
      "A Xiaomi é uma das maiores fabricantes de celulares do mundo",
      "O campeonato de basquete está emocionante"
]
categorias = ["tecnologia", "esportes", "política", "tecnologia", "esportes", "política", "política", "política", "tecnologia", "esportes"]
  
# Convertendo textos em uma matriz de contagens de tokens
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)
  
# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, categorias, test_size=0.5, random_state=42)
  
# Treinando o classificador
clf = MultinomialNB()
clf.fit(X_train, y_train)
  
# Predição e Avaliação
y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")

# Classificando um novo texto
novo_texto = ["A Ucrânia está em guerra com a Rússia"]
novo_texto_vetorizado = vectorizer.transform(novo_texto)
predicao = clf.predict(novo_texto_vetorizado)
print(f"O texto '{novo_texto[0]}' pertence à categoria: {predicao[0]}")