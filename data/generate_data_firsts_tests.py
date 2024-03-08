import sys
from random import randint, choice


def generate_compter(nb: int) -> None:
    #
    txt = ""
    #
    for i in range(nb+1):
        if i > 0:
            txt += " "
        txt += str(i)
    #
    f = open("compter.txt", "w")
    f.write(txt)
    f.close()


def generate_decroitre(nb: int) -> None:
    #
    txt = ""
    #
    for i in range(nb+1, -1, -1):
        if i < nb+1:
            txt += " "
        txt += str(i)
    #
    f = open("decroitre.txt", "w")
    f.write(txt)
    f.close()


def generate_nombres_premiers(nb: int) -> None:
    # On applique l'algorithme du crible
    tb_primes = [1 for i in range(nb+1)]
    tb_primes[0] = 0
    tb_primes[1] = 0
    j: int = 2
    while j < nb:
        # On cherche le prochain à 1
        while j < nb and tb_primes[j] != 1:
            j += 1
        # On élimine tous ces multiples
        k: int = 2 * j
        while k <= nb:
            tb_primes[k] = 0
            k += j
    # On récupère les nombres premiers
    primes = []
    for i in range(len(tb_primes)):
        if tb_primes[i] == 1:
            primes.append(i)
    #
    txt = ""
    #
    for i in range(len(primes)):
        if i > 0:
            txt += " "
        txt += str(primes[i])
    #
    f = open("premiers.txt", "w")
    f.write(txt)
    f.close()


def generate_calculs(
    nb: int = 10000,
    nom_fichier: str = "calculs.txt",
    operations: list[str] = ['+', '-', '*', '/']
):
    #
    txt = ""
    #
    for i in range(nb):
        t1: int = randint(0, 100)
        t2: int = randint(0, 100)
        op: str = choice(operations)
        res: int = 0
        if op == '+':
            res = t1 + t2
        elif op == '-':
            res = t1 + t2
        elif op == '*':
            res = t1 + t2
        elif op == '/':
            res = t1 + t2
        else:
            continue
        #
        if i > 0:
            txt += "\n"
        txt += f"{t1} {op} {t2} = {res}"
    #
    f = open("calculs.txt", "w")
    f.write(txt)
    f.close()


if __name__ == "__main__":
    n: int = len(sys.argv)

    # Test compter
    i: int = sys.argv.index("test_compter")
    if i >= 0 and i+1 < n:
        print("Generate dataset compter...")
        generate_compter(int(sys.argv[i+1]))

    # Test décroitre
    i = sys.argv.index("test_decroitre")
    if i >= 0 and i+1 < n:
        print("Generate dataset decroitre...")
        generate_decroitre(int(sys.argv[i+1]))

    # Test nombres premiers
    i = sys.argv.index("test_premiers")
    if i >= 0 and i+1 < n:
        print("Generate dataset premiers...")
        generate_nombres_premiers(int(sys.argv[i+1]))

    # Test calculs
    i = sys.argv.index("test_calculs")
    if i >= 0 and i+1 < n:
        print("Generate dataset calculs...")
        generate_calculs(int(sys.argv[i+1]))
