import sys
import time

def update_console(text):
	sys.stdout.write("\r" + text)
	sys.stdout.flush() # important
def end_console():
	print("\n")
for it in range(100):
	update_console("Iteration:" + str(float(it)/100.0))
	time.sleep(0.01)
print("\nEnd")
