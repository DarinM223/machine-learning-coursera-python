from app.ex1 import ex1_multi
from app.ex2 import ex2, ex2_reg
import sys

arguments = sys.argv

optional_argument = False
for arg in arguments:
    if arg == 'ex1':
        ex1_multi.run()
        optional_argument = True
    elif arg == 'ex2':
        ex2.run()
        optional_argument = True
    elif arg == 'ex2_reg':
        ex2_reg.run()
        optional_argument = True

if optional_argument == False:
    # Run web application
    from app.webapp import app
    if __name__ == '__main__':
        app.run(debug=True)
