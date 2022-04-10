CC = gcc
CFLAGS = -Wall

all: compile
	./results results.csv

compile:
	$(CC) $(CFLAGS) ./main.c -o ./results -L/usr/local/lib -lgsl -lgslcblas

clean:
	rm -rf results results.csv
