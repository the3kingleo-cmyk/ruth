CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Wno-misleading-indentation -D_POSIX_C_SOURCE=200809L

ruth-core: ruth_core.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f ruth-core
.PHONY: clean
