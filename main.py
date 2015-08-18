from app.ex1 import ex1_multi
import sys

arguments = sys.argv

optional_argument = False
for arg in arguments:
    if arg == 'ex1':
        ex1_multi.run()
        optional_argument = True

if optional_argument != True:
    # Run web application
    from app.webapp import app
    if __name__ == '__main__':
        app.run(debug=True)
