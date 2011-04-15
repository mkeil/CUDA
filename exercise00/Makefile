OBJ := main.o liste.o

all: $(OBJ)
	gcc -o program $+
	
%.o: %.c
	gcc -c $<
	
.PHONY: clean
clean:
	rm -rf *.o