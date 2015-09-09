from app.ex1 import ex1_multi
from app.ex2 import ex2, ex2_reg
from app.ex3 import ex3
import sys

arguments = sys.argv

modules = {
    'ex1': ex1_multi,
    'ex2': ex2,
    'ex2_reg': ex2_reg,
    'ex3': ex3
}

optional_argument = False
for arg in arguments:
    if arg in modules:
        modules[arg].run()
        optional_argument = True

if optional_argument == False:
    # Run web application
    from app.webapp import app
    if __name__ == '__main__':
        app.run(debug=True)
