---
jupytext:
  cell_metadata_json: true
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
nbhosting:
  title: suite du TP simple avec des images
---

Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

+++ {"slideshow": {"slide_type": ""}, "tags": []}

# TP images (2/2)

merci à Wikipedia et à stackoverflow

**le but de ce TP n'est pas d'apprendre le traitement d'image  
on se sert d'images pour égayer des exercices avec `numpy`  
(et parce que quand on se trompe ça se voit)**

```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt
```

+++ {"tags": ["framed_cell"]}

````{admonition} → **notions intervenant dans ce TP**

* sur les tableaux `numpy.ndarray`
  * `reshape()`, masques booléens, *ufunc*, agrégation, opérations linéaires
  * pour l'exercice `patchwork`:  
    on peut le traiter sans, mais l'exercice se prête bien à l'utilisation d'une [indexation d'un tableau par un tableau - voyez par exemple ceci](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
  * pour l'exercice `sepia`:  
    ici aussi on peut le faire "naivement" mais l'utilisation de `np.dot()` peut rendre le code beaucoup plus court
* pour la lecture, l'écriture et l'affichage d'images
  * utilisez `plt.imread()`, `plt.imshow()`
  * utilisez `plt.show()` entre deux `plt.imshow()` si vous affichez plusieurs images dans une même cellule

  ```{admonition} **note à propos de l'affichage**
  :class: seealso dropdown admonition-small

  * nous utilisons les fonctions d'affichage d'images de `pyplot` par souci de simplicité
  * nous ne signifions pas là du tout que ce sont les meilleures!  
    par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
    alors que la fonction `save` de `PIL` le permet
  * vous êtes libres d'utiliser une autre librairie comme `opencv`  
    si vous la connaissez assez pour vous débrouiller (et l'installer), les images ne sont qu'un prétexte...
  ```
````

+++

## Création d'un patchwork

+++

1. Le fichier `data/rgb-codes.txt` contient une table de couleurs:
```
AliceBlue 240 248 255
AntiqueWhite 250 235 215
Aqua 0 255 255
.../...
YellowGreen 154 205 50
```
Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.

```{code-cell} ipython3
# votre code
colors={}
with open('data/rgb-codes.txt', 'r') as file:
    for line in file:
        colname, *L = line.split()
        colors[colname] = np.array([int(e) for e in L], dtype=np.uint8)
colors
```

2. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
`'Red'`, `'Lime'`, `'Blue'`

```{code-cell} ipython3
# votre code
print(colors['Red'], colors['Lime'], colors['Blue'])
```

3. Faites une fonction `patchwork` qui  

   * prend une liste de couleurs et la structure donnant le code des couleurs RGB
   * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
   * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
   si besoin de compléter l'image mettez du blanc

+++

````{admonition} indices
:class: dropdown
  
* sont potentiellement utiles pour cet exo:
  * la fonction `np.indices()`
  * [l'indexation d'un tableau par un tableau](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
* aussi, ça peut être habile de couper le problème en deux, et de commencer par écrire une fonction `rectangle_size(n)` qui vous donne la taille du patchwork en fonction du nombre de couleurs  
  ```{admonition} et pour calculer la taille au plus juste
  :class: tip dropdown

  en version un peu brute, on pourrait utiliser juste la racine carrée;
  par exemple avec 5 couleurs créer un carré 3x3 - mais 3x2 c'est quand même mieux !

  voici pour vous aider à calculer le rectangle qui contient n couleurs

  n | rect | n | rect | n | rect | n | rect |
  -|-|-|-|-|-|-|-|
  1 | 1x1 | 5 | 2x3 | 9 | 3x3 | 14 | 4x4 |
  2 | 1x2 | 6 | 2x3 | 10 | 3x4 | 15 | 4x4 |
  3 | 2x2 | 7 | 3x3 | 11 | 3x4 | 16 | 4x4 |
  4 | 2x2 | 8 | 3x3 | 12 | 3x4 | 17 | 4x5 |
  ```
````

```{code-cell} ipython3
# votre code
def rectangle_size(n):  #on prend les deux plus grand diviseurs de n
    p=int(np.sqrt(n))
    q=n
    while n%p!=0 and p!=1:
        p-=1
    q=n//p
    return (p,q)


def patchwork(couleurs):
    p,q=rectangle_size(len(couleurs))
    colo=np.array([colors[e] for e in couleurs])
    pattern = np.arange(0, p*q).reshape(p, q)
    return colo[pattern]
```

4. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.

```{code-cell} ipython3
# votre code
couleurs = np.random.choice(list(colors.keys()), 16)
#print(couleurs)
plt.imshow(patchwork(couleurs))
```

5. Sélectionnez toutes les couleurs à base de blanc et affichez leur patchwork  
même chose pour des jaunes

```{code-cell} ipython3
# votre code
patch1=[]
for x in colors.keys() :
    if "White" in x:
        patch1.append(x)
plt.imshow(patchwork(patch1))
plt.show()
patch2=[]
for x in colors.keys() :
    if "Yellow" in x:
        patch2.append(x)
plt.imshow(patchwork(patch2))
plt.show()
```

6. Appliquez la fonction à toutes les couleurs du fichier  
et sauver ce patchwork dans le fichier `patchwork.png` avec `plt.imsave`

```{code-cell} ipython3
# votre code
couleurs_all = np.random.choice(list(colors.keys()), len(colors))
image = patchwork(couleurs_all)
plt.imsave('patchwork.png', image)
```

7. Relisez et affichez votre fichier  
   attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels

vous devriez obtenir quelque chose comme ceci

```{image} media/patchwork-all.jpg
:align: center
```

```{code-cell} ipython3
# votre code
fichier = plt.imread("patchwork.png")
plt.imshow(fichier)
```

## Somme dans une image & overflow

+++

0. Lisez l'image `data/les-mines.jpg`

```{code-cell} ipython3
# votre code
img = plt.imread("data/les-mines.jpg")
```

1. Créez un nouveau tableau `numpy.ndarray` en sommant **avec l'opérateur `+`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
# votre code
new1 = img[:,:,0]+img[:,:,1]+img[:,:,2]
```

2. Regardez le type de cette image-somme, et son maximum; que remarquez-vous?  
   Affichez cette image-somme; comme elle ne contient qu'un canal il est habile de l'afficher en "niveaux de gris" (normalement le résultat n'est pas terrible ...)


   ```{admonition} niveaux de gris ?
   :class: dropdown tip

   cherchez sur google `pyplot imshow cmap gray`
   ```

```{code-cell} ipython3
# votre code
print(new1.dtype,new1.max())

plt.imshow(new1, cmap='gray')
```

3. Créez un nouveau tableau `numpy.ndarray` en sommant mais cette fois **avec la fonction d'agrégation `np.sum`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
# votre code
new2 = img.sum(axis=2)
```

4. Comme dans le 2., regardez son maximum et son type, et affichez la

```{code-cell} ipython3
# votre code
print(new2.dtype,new2.max())

plt.imshow(new2, cmap='gray')
```

5. Les deux images sont de qualité très différente, pourquoi cette différence ? Utilisez le help `np.sum?`

```{code-cell} ipython3
# votre code / explication
help(np.sum)
#le calcul avec np.sum passe automatiquement en uint64 si jamais la valeur dépasse 255
#ce qui permet d'éviter le seuil. Avec + le tableau reste en uint8.
```

6. Passez l'image en niveaux de gris de type entiers non-signés 8 bits  
(de la manière que vous préférez)

```{code-cell} ipython3
# votre code
new3=new2.copy()
new3.dtype=np.uint8
plt.imshow(new3, cmap='gray')
```

7. Remplacez dans l'image en niveaux de gris,  
les valeurs >= à 127 par 255 et celles inférieures par 0  
Affichez l'image avec une carte des couleurs des niveaux de gris  
vous pouvez utilisez la fonction `numpy.where`

```{code-cell} ipython3
# votre code
tab = new2>=127
img2=255*tab

plt.imshow(img2, cmap='gray')
```

8. avec la fonction `numpy.unique`  
regardez les valeurs différentes que vous avez dans votre image en noir et blanc

```{code-cell} ipython3
# votre code
np.unique(img2)
```

## Image en sépia

+++

Pour passer en sépia les valeurs R, G et B d'un pixel  
(encodées ici sur un entier non-signé 8 bits)  

1. on transforme les valeurs `R`, `G` et `B` par la transformation  
`0.393 * R + 0.769 * G + 0.189 * B`  
`0.349 * R + 0.686 * G + 0.168 * B`  
`0.272 * R + 0.534 * G + 0.131 * B`  
(attention les calculs doivent se faire en flottants pas en uint8  
pour ne pas avoir, par exemple, 256 devenant 0)  
1. puis on seuille les valeurs qui sont plus grandes que `255` à `255`
1. naturellement l'image doit être ensuite remise dans un format correct  
(uint8 ou float entre 0 et 1)

+++

````{tip}
jetez un coup d'oeil à la fonction `np.dot` 
qui est si on veut une généralisation du produit matriciel

dont voici un exemple d'utilisation:
````

```{code-cell} ipython3
:scrolled: true

# exemple de produit de matrices avec `numpy.dot`
# le help(np.dot) dit: dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])

i, j, k, m, n = 2, 3, 4, 5, 6
A = np.arange(i*j*k).reshape(i, j, k)
B = np.arange(m*k*n).reshape(m, k, n)

C = A.dot(B)
# or C = np.dot(A, B)

print(f"en partant des dimensions {A.shape} et {B.shape}")
print(f"on obtient un résultat de dimension {C.shape}")
print(f"et le nombre de termes dans chaque `sum()` est {A.shape[-1]} == {B.shape[-2]}")
```

**Exercice**

+++

1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
la fonction `numpy.dot` peut être utilisée si besoin, voir l'exemple ci-dessus

```{code-cell} ipython3
# votre code
def sepia(image):
    #on commence par changer le type 
    im = image.astype('float')
    #puis on sépare les 3 canaux
    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]
    #on fait les calculs
    Sr = 0.393 * R + 0.769 * G + 0.189 * B
    Sg = 0.349 * R + 0.686 * G + 0.168 * B
    Sb = 0.272 * R + 0.534 * G + 0.131 * B
    #on seuil les valeurs, j'ai consulté le help de np.where pour l'utiliser
    r1 = np.where(Sr<=255,Sr,255)
    g1 = np.where(Sg<=255,Sg,255)
    b1 = np.where(Sb<=255,Sb,255)
    #on remet en une image
    new_im = np.zeros(image.shape)
    new_im[:,:,0]=r1
    new_im[:,:,1]=g1
    new_im[:,:,2]=b1
    #on change le type
    imf = new_im.astype('uint8')
    return imf
```

2. Passez votre patchwork de couleurs en sépia  
Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso

```{code-cell} ipython3
# votre code
file = plt.imread("patchwork-all.jpg")
plt.imshow(sepia(file))
```

3. Passez l'image `data/les-mines.jpg` en sépia

```{code-cell} ipython3
# votre code
mine = plt.imread("data/les-mines.jpg")
plt.imshow(sepia(mine))
```

## Exemple de qualité de compression

+++

1. Importez la librairie `Image`de `PIL` (pillow)  
(vous devez peut être installer PIL dans votre environnement)

```{code-cell} ipython3
# votre code
from PIL import Image
```

2. Quelle est la taille du fichier `data/les-mines.jpg` sur disque ?

```{code-cell} ipython3
file = "data/les-mines.jpg"
```

```{code-cell} ipython3
# votre code
im1 = Image.open(file)
print(im1.size)
```

3. Lisez le fichier 'data/les-mines.jpg' avec `Image.open` et avec `plt.imread`

```{code-cell} ipython3
# votre code
im2 = plt.imread(file)
```

4. Vérifiez que les valeurs contenues dans les deux objets sont proches

```{code-cell} ipython3
# votre code
im2.shape
```

5. Sauvez (toujours avec de nouveaux noms de fichiers)  
l'image lue par `imread` avec `plt.imsave`  
l'image lue par `Image.open` avec `save` et une `quality=100`  
(`save` s'applique à l'objet créé par `Image.open`)

```{code-cell} ipython3
# votre code
plt.imsave('im2.jpg',im2)
im1.save('im1.jpg', quality = 100)
```

6. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
Que constatez-vous ?

```{code-cell} ipython3
# votre code
#im1 a une taille de 559.3 KB
#im2 a une taille de 133.5 KB
#La qualité de im1 est meilleure (plus nette)
```

7. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence

```{code-cell} ipython3
# votre code
im11 = plt.imread('im1.jpg')
im22 = plt.imread('im2.jpg')

plt.figure(1)
plt.imshow(im11)
plt.figure(2)
plt.imshow(im22)
plt.show()
```

```{code-cell} ipython3

```
