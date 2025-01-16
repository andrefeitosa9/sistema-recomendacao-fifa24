
Sistema de Recomendação de Jogadores

Link para acessar os dados brutos iniciais no Kaggle - aqui - https://www.kaggle.com/datasets/joebeachcapital/fifa-players/data

Repositório base para artigo publicado no Medium - E se as contratações fossem baseadas somente em ratings? - Sistemas de Recomendação em busca de jogadores similares
Link - https://medium.com/@andrefeitosa9/e-se-as-contratações-fossem-baseadas-somente-em-ratings-669d1773abd6

Este é um projeto de Sistema de Recomendação de Jogadores, criado com o objetivo de ajudar a identificar jogadores que se encaixem em requisitos específicos, usando técnicas de aprendizado de máquina e análise de dados.

🚀 Funcionalidades

Recomendações personalizadas: Sugere jogadores com características similares com base em critérios de rating no jogo FIFA 24.

Análise de similaridade: Utiliza algoritmos como KNN (K-Nearest Neighbors) e similaridade de cosseno para identificar jogadores similares.

📊 Tecnologias Utilizadas

Linguagem principal: Python

Bibliotecas e frameworks principais:

Pandas para análise de dados

Scikit-learn para implementação dos algoritmos de aprendizado de máquina (KNN e cálculo da similaridade de cosseno)

Matplotlib e Seaborn para visualizações gráficas

🗂 Estrutura do Projeto

📁 sistema-recomendacao-jogadores

├── 📁 dados

│   ├── jogadores_fifa_2024_bruto.csv  # Dataset principal

├── 📁 recomendacoes                   # Pasta com recomendacoes

│   ├── ...      

├── requirements.txt                   # Dependências do projeto

└── README.md                          # Documentação do projeto


🔧 Como Configurar o Projeto

Clone o repositório:

git clone https://github.com/andrefeitosa9/sistema-recomendacao-fifa24.git
cd sistema-recomendacao-jogadores

Crie e ative um ambiente virtual:

python -m venv venv
source venv/bin/activate  # No Windows, use venv\Scripts\activate

Instale as dependências:

pip install -r requirements.txt

Adicione o dataset:
Certifique-se de que o arquivo jogadores_fifa_2024_bruto.csv esteja na pasta dados.

🔧 Como Contribuir

Realize um fork do repositório

Crie uma branch para sua funcionalidade: git checkout -b minha-funcionalidade

Realize commit das suas alterações: git commit -m 'Adiciona nova funcionalidade'

Envie para sua branch: git push origin minha-funcionalidade

Abra um Pull Request

💡 Considerações Finais

Contribuições são muito bem-vindas! Fique à vontade para explorar, sugerir melhorias ou reportar problemas.

🔗 Links

Dados brutos iniciais no Kaggle - https://www.kaggle.com/datasets/joebeachcapital/fifa-players/data

Repositório no GitHub - https://github.com/andrefeitosa9/sistema-recomendacao-fifa24

Artigo no Medium: "E se as contratações fossem baseadas somente em ratings? – Sistemas de Recomendação em busca de jogadores similares"

