import random
import string

dna = set(string.printable)

class Sequence(str):

    def mutate(self, d, n):
        mutants = set([self])
        while len(mutants) < n:
            k = random.randint(1, d)
            for _ in range(k):
                mutant_type = random.choice(["d", "s", "i"])
                if mutant_type == "i":
                    mutants.add(self.insertion(k))
                elif mutant_type == "d":
                    mutants.add(self.deletion(k))
                elif mutant_type == "s":
                    mutants.add(self.substitute(k))
        return list(mutants)


    def deletion(self, n):
        if n >= len(self):
            return ""
        chars = list(self)
        i = 0
        while i < n:
            idx = random.choice(range(len(chars)))
            del chars[idx]
            i += 1
        return "".join(chars)

    def insertion(self, n):
        chars = list(self)
        i = 0
        while i < n:
            idx = random.choice(range(len(chars)))
            new_base = random.choice(list(dna))
            chars.insert(idx, new_base)
            i += 1
        return "".join(chars)

    def substitute(self, n):
        idxs = random.sample(range(len(self)), n)
        chars = list(self)
        for i in idxs:
            new_base = random.choice(list(dna.difference(chars[i])))
            chars[i] = new_base
        return "".join(chars)
    
   
# TEST AREA #
if __name__ == "__main__":
    s0=['HP C6567B Coated Paper 1 roll 42-inches x 150 ft', 
        'Stationery &amp; Office Machinery', 'HP', 'C6567B', '69.88']
    for i in range(len(s0)):
        s=Sequence(s0[i])
        #s = Sequence("AAAAA")
        d = 1  # max edit distance
        n = 2  # number of strings in result
        mutates=s.mutate(d, n)
        print(mutates[1])
    #>>> ['AAA', 'GACAAAA', 'AAAAA', 'CAGAA', 'AACAAAA']'