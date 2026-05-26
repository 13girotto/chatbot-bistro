import numpy as np
import re
import unicodedata
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1. FUNÇÃO DE PRÉ-PROCESSAMENTO (PLN) - UNIFICADA

def limpar_e_tokenizar(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.split()

# 2. BASE DE DADOS (DATASET SUPERVISIONADO)

respostas = {
    "horario": "Funcionamos de segunda a sexta, das 18h45 às 22h.",
    "reservas": "Reservas apenas para grupos a partir de 5 pessoas. Menos que isso, é por ordem de chegada.",
    "delivery": "Não fazemos entregas diretas. Pedidos via WhatsApp com retirada no local ou pelos apps de entrega parceiros.",
    "restricoes": "Não aceitamos Pets, não aceitamos Vales-Refeição.",
    "localizacao": "Ficamos na QSD 12, Taguatinga Sul.",
    "cardapio": "Nosso menu é focado em uma excelente Casa de Risotos e Massas!",
    "saudacao": "Olá! Você entrou em contato com o @quintaldicasabistro. Como podemos te ajudar hoje?",
    "rolha": "No Quintal Di'Casa você pode trazer o seu próprio vinho! Cobramos uma taxa de R$ 50,00 por garrafa aberta. Nós fornecemos as taças adequadas e o serviço de balde com gelo para manter a temperatura ideal. Solicite também nossa carta de vinhos",
    "bebidas": "Temos opções de refrigerantes, sucos da polpa, cervejas trincando e uma excelente carta de vinhos!",
}

dados_treino = [
    # --- SAUDAÇÃO ---
    ("oi", "saudacao"),
    ("ola", "saudacao"),
    ("bom dia", "saudacao"),
    ("boa tarde", "saudacao"),
    ("boa noite", "saudacao"),
    
    # --- HORÁRIO ---
    ("que horas voces abrem", "horario"),
    ("esta aberto hoje", "horario"),
    ("voces estao abertos", "horario"),
    ("horario", "horario"),
    ("funcionamento", "horario"),
    ("abre que horas", "horario"),
    ("que horas voces fecham", "horario"),
    
    # --- RESERVAS ---
    ("quero reservar uma mesa", "reservas"),
    ("vcs aceitam reserva", "reservas"),
    ("reservas", "reservas"),
    ("reserva", "reservas"),
    ("tem reservas", "reservas"),
    ("fazer reserva", "reservas"),
    ("marcar mesa", "reservas"),
    
    # --- DELIVERY / PEDIDOS ---
    ("voces fazem entrega", "delivery"),
    ("qual o link do ifood", "delivery"),
    ("entrega", "delivery"),
    ("delivery", "delivery"),
    ("quero fazer um pedido", "delivery"),
    ("fazer pedido", "delivery"),
    ("pedir comida", "delivery"),
    ("como eu peco", "delivery"),
    ("pedido", "delivery"),
    ("posso fazer um pedido para pegar ai", "delivery"),
    
    # --- REGRAS / RESTRIÇÕES ---
    ("aceita pet cachorro", "restricoes"),
    ("voces aceitam vale refeicao sodexo alelo", "restricoes"),
    ("vr va", "restricoes"),
    ("regras", "restricoes"),
    ("restricoes", "restricoes"),
    
    # --- LOCALIZAÇÃO ---
    ("onde fica o restaurante", "localizacao"),
    ("qual o endereco de vcs", "localizacao"),
    ("localizacao", "localizacao"),
    ("local", "localizacao"),
    ("onde fica", "localizacao"),
    ("endereco", "localizacao"),
    
    # --- CARDÁPIO ---
    ("cardapio", "cardapio"),
    ("menu", "cardapio"),
    ("quais sao os pratos", "cardapio"),
    ("oque tem para comer", "cardapio"),

    # --- ROLHA ---
    ("posso levar meu vinho", "rolha"),
    ("qual o valor da taxa de rolha", "rolha"),
    ("voces cobram para levar bebida", "rolha"),
    ("taxa de rolha", "rolha"),
    ("levar vinho de casa", "rolha"),
    ("posso levar meu proprio vinho", "rolha"),
    ("valor da rolha", "rolha"),
    ("quanto cobra para abrir garrafa", "rolha"),

    # --- BEBIDAS ---
    ("quais as opcoes de refrigerante e suco", "bebidas"),
    ("o que tem para beber", "bebidas"),
    ("voces tem coca cola refrigerante", "bebidas"),
    ("tem suco ou cerveja", "bebidas"),
    ("bebidas", "bebidas"),
]

todas_as_palavras = []
intencoes = list(respostas.keys())

for frase, intencao in dados_treino:
    todas_as_palavras.extend(limpar_e_tokenizar(frase))
todas_as_palavras = sorted(list(set(todas_as_palavras)))

def vetorizar_texto(texto, vocabulo):
    palavras_texto = limpar_e_tokenizar(texto)
    vetor = np.zeros(len(vocabulo))
    for i, pal in enumerate(vocabulo):
        if pal in palavras_texto:
            vetor[i] = 1
    return vetor

X_treino = np.array([vetorizar_texto(frase, todas_as_palavras) for frase, _ in dados_treino])
Y_treino = np.array([intencoes.index(intencao) for _, intencao in dados_treino])

# 3. REDE NEURAL ARTIFICIAL (DO ZERO)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

np.random.seed(42)
pesos = np.random.randn(len(todas_as_palavras), len(intencoes)) * 0.01

taxa_aprendizado = 0.1
for epoca in range(800):  
    saida_linear = np.dot(X_treino, pesos)
    predicoes = softmax(saida_linear)
    
    Y_um_quente = np.zeros_like(predicoes)
    Y_um_quente[np.arange(len(Y_treino)), Y_treino] = 1
    erro = predicoes - Y_um_quente
    
    gradiente = np.dot(X_treino.T, erro)
    pesos -= taxa_aprendizado * gradiente

print("IA do Bistrô treinada com sucesso!")
print("Servidor Flask iniciando...\n")

# 4. ENDPOINT DA API (WEB / WHATSAPP)
@app.route("/chatbot", methods=["POST"])
def receber_mensagem():
    dados = request.get_json()
    
    if not dados or "mensagem" not in dados:
        return jsonify({"erro": "Formato inválido. Envie um JSON com o campo 'mensagem'."}), 400
        
    mensagem_cliente = dados.get("mensagem")
    
    vetor_entrada = vetorizar_texto(mensagem_cliente, todas_as_palavras)

    if np.sum(vetor_entrada) == 0:
        resposta_erro = ("Desculpe, não consegui entender. Pode reescrever de outra forma? "
                         "Pode perguntar sobre: horário, reservas, entregas, restrições ou cardápio.")
        return jsonify({"resposta": resposta_erro})
        
    resultado_rede = np.dot(vetor_entrada, pesos)
    probabilidades = softmax(resultado_rede)    
    
    indice_final = np.argmax(probabilidades)
    intencao_prevista = intencoes[indice_final]
    confianca = probabilidades[indice_final]

    if confianca > 0.35: 
        return jsonify({"resposta": respostas[intencao_prevista]})
    else:
        return jsonify({"resposta": "Fiquei na dúvida sobre o seu pedido. Pode detalhar melhor sua dúvida?"})

if __name__ == "__main__":
    import os
    porta = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=porta, debug=False)