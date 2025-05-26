from pathlib import Path
from dotenv import load_dotenv

# Caminho absoluto para a raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent

# Carrega variáveis de ambiente
load_dotenv(PROJECT_ROOT / '.env')

# Caminhos importantes
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_CSV = DATA_DIR / 'input.csv'
OUTPUT_DIR = DATA_DIR / 'output'