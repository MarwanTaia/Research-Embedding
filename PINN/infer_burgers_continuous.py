'''
FICHIER:
--------
    infer_burg.py

DESCRIPTION:
------------
Ce fichier contient le programme de réimplémentation de PINN (Physics Informed
    Neural Network) pour la résolution de l'équation de Burgers.
Dans notre cas x est la position et t le temps. Chacun des deux est un vecteur,
    donc unidimensionnel. Le problème est défini sur l'intervalle [0,1] pour x,
    et [0,1] pour t.
La condition initiale est u(x,0) = -sin(pi*x) et la condition aux limites est
    u(0,()) = u(1,t) = 0.
'''

################################################################################
# IMPORTS
################################################################################
import numpy as np 
import tensorflow as tf 
import scipy.io as sio # Pour charger les données
from pyDOE import lhs # Latin Hypercube Sampling
from scipy.interpolate import griddata # Interpolation
import matplotlib.pyplot as plt 

################################################################################
# MACROS
################################################################################
# Chemin relatif vers le fichier de données
DATA_PATH = 'data/burgers_shock.mat'

# Macros pour les affichages
VERIF_PLOTS = True
PLOTS_3D = True
PLOTS = True

# Marcos pour l'apprentissage
EPOCHS = 30
LEARING_RATE = 0.001
# Nombre de couches et de neurones par couche
# Ici on a 2 entrées (x et t) et 1 sortie (u)
LAYERS = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1]

################################################################################
# CLASSES
################################################################################
class PINN:
    '''
    Classe PINN
    -----------
    Cette classe contient les méthodes permettant de construire et d'entraîner
        un PINN pour la résolution de l'équation de Burgers.
    
    Dans cette classe, on définit les méthodes:
    - __init__
    - net_u
    - net_f
    - make_NN
    - loss_func
    - callback

    Note : dans cette classe, on créé le réseau de neurones directement dans
        la méthode make_NN, et on l'entraîne dans la méthode train. Tout
        directement avec TensorFlow.
    '''

    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
        '''
        Méthode __init__
        ----------------
        Cette méthode initialise les attributs de la classe PINN.
        '''
        self.lb = lb # borne inférieure de la grille
        self.ub = ub # borne supérieure de la grille

        # On récupère les coordonnées des points de la condition aux limites
        #  et de la condition initiale.
        self.x_u = X_u[:,0:1] # coordonnées x des points de la condition aux limites
        self.t_u = X_u[:,1:2] # coordonnées t des points de la condition aux limites

        # On récupère les coordonnées des points de la fonction f.
        self.x_f = X_f[:,0:1] # coordonnées x des points de la fonction f
        self.t_f = X_f[:,1:2] # coordonnées t des points de la fonction f

        # On récupère les valeurs de la condition aux limites et de la condition
        #  initiale.
        self.u = u # valeurs de la solution aux points de la condition aux limites

        self.layers = layers # Liste contenant le nombre de neurones par couche
        self.nu = nu # coefficient de diffusion
        
        # On construit le réseau de neurones
        self.neuralNet = self.make_NN(layers)

        self.loss_history = [] # Liste contenant l'historique des erreurs durant l'entraînement

    def net_u(self, x, t):
        '''
        METHODE:
        --------
            net_u
        
        DESCRIPTION:
        ------------
            Cette méthode rend la prédiction de la solution u(x,t) par le réseau
                de neurones.
        
        PARAMETRES:
        -----------
            x : Tenseur contenant les coordonnées x des points où on veut
                prédire la solution.
            t : Tenseur contenant les coordonnées t des points où on veut
                prédire la solution.

        RETOUR:
        -------
            u : Tenseur contenant la prédiction de la solution aux points
                (x,t).
        '''
        return self.neuralNet(tf.concat([x,t],1))

    def net_f(self, x, t):
        '''
        METHODE:
        --------
            net_f

        DESCRIPTION:
        ------------
            Cette méthode rend la prédiction de la fonction f(x,t) avec les
                prédiction de la solution u(x,t) et de sa dérivée partielle
                par rapport à x et par rapport à t.

        PARAMETRES:
        -----------
            x : Tenseur ou array contenant les coordonnées x des points où on
                veut prédire la fonction f.
            t : Tenseur ou array contenant les coordonnées t des points où on
                veut prédire la fonction f.

        RETOUR:
        -------
            f : Tenseur contenant la prédiction de la fonction f aux points
                (x,t).
        '''
        # On convertit x et t en tenseurs
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            u = self.net_u(x,t)
            u_x = tape.gradient(u, x)
            u_xx = tape.gradient(u_x, x)
            u_t = tape.gradient(u, t)

        del tape

        return u_t + u*u_x - self.nu*u_xx

    def make_NN(self, layers):
        '''
        METHODE:
        --------
            make_NN
        
        DESCRIPTION:
        ------------
            Cette méthode construit le réseau de neurones.

        PARAMETRES:
        -----------
            layers : Liste contenant le nombre de neurones par couche.
                layers[0] = nombre d'entrées
                layers[-1] = nombre de sorties
                layers[1:-1] = nombre de neurones par couche cachée

        RETOUR:
        -------
            neuralNet : Réseau de neurones construit.
        '''
        # On définit le modèle
        model = tf.keras.models.Sequential()
        # On ajoute la première couche cachée
        model.add(tf.keras.layers.Dense(layers[1], input_dim=layers[0], activation='tanh', kernel_initializer='glorot_normal', dtype=tf.float32)) # TOTO ACTIVAT
        # On ajoute les autres couches cachées
        for i in range(2, len(layers)-1):
            model.add(tf.keras.layers.Dense(layers[i], activation='tanh', kernel_initializer='glorot_normal'))
        # On ajoute la couche de sortie
        model.add(tf.keras.layers.Dense(layers[-1], activation='linear', kernel_initializer='glorot_normal'))

        return model

    def loss_func(self,a,b):
        '''
        METHODE:
        --------
            loss_func

        DESCRIPTION:
        ------------
            Cette méthode rend la fonction de perte du réseau de neurones,
                associée à la méthode PINN au problème de Burgers, pour
                l'entraînement.
        '''
        # On calcule la prédiction de la solution aux points de la condition
        #  aux limites.
        u_pred = self.net_u(self.x_u, self.t_u)
        # On calcule la prédiction de la fonction f aux points de la fonction f.
        f_pred = self.net_f(self.x_f, self.t_f)

        # On calcule la fonction de perte
        loss_u = tf.reduce_mean(tf.square(u_pred - self.u))
        loss_f = tf.reduce_mean(tf.square(f_pred))

        # On rend la somme des deux membres de la fonction de perte.
        return loss_u + loss_f

    def callback(self, loss):
        '''
        Méthode callback
        ----------------
        Cette méthode permet d'afficher la valeur de la fonction de coût à
            chaque itération.
        '''
        #print('Losses: %.3e' % (loss))

    def train(self, epochs, batch_size=1, learning_rate=1e-3, patience=5, verbose=True):
        '''
        Méthode train
        -------------
        Cette méthode entraîne le réseau de neurones.
        Avec patience = 10, on arrête l'entraînement si la fonction de coût
            n'a pas diminué pendant 10 époques.
        '''
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        loss_history = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.callback(logs['loss']))
        print(loss_history)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)

        self.neuralNet.compile(optimizer, loss=self.loss_func, run_eagerly=True)

        self.neuralNet.fit(x=np.concatenate((self.x_u, self.t_u), 1), y=self.u, batch_size=1, epochs=epochs, callbacks=[loss_history, early_stopping], verbose=verbose)

    def predict(self, X_star):
        '''
        Méthode predict
        ---------------
        Cette méthode rend la prédiction de la solution u(x,t) par le PINN.
        '''
        f_pred = self.net_f(X_star[:,0:1], X_star[:,1:2])
        u_pred = self.net_u(X_star[:,0:1], X_star[:,1:2])

        return u_pred, f_pred

################################################################################
# MAIN
################################################################################
if __name__ == "__main__":
    ############################################################################
    # PARAMETRES
    ############################################################################
    nu = 0.01/np.pi # coefficient de diffusion
    noise = 0.0 # bruit ajouté aux données # TODO: RECHECK

    N_u = 100 # Nombre de points de données pour l'entrainement de u sur les conditions aux limites (t=0 ou x=0 et x=1)
    N_f = 10000 # Nombre de points de données pour l'entrainement de f (colocation points) (sur t=0 et t=1)

    ############################################################################
    # DATA INITIALIZATION
    ############################################################################
    # Importation des données
    data = sio.loadmat(DATA_PATH)

    # Affichage des données
    print("\nData : " + '-'*80)
    ## List of variables
    print("List of variables: ", data.keys())
    ## Shape of variables
    print("Shape of x: ", data['x'].shape)
    print("Shape of t: ", data['t'].shape)
    print("Shape of usol: ", data['usol'].shape)
    print('')

    # Extraction des données
    x = data['x'].flatten()[:,None] # x est un vecteur de 256 éléments. Donc unidimensionnel.
    t = data['t'].flatten()[:,None] # t est un vecteur de 100 éléments. Donc unidimensionnel.
    usol = np.real(data['usol']).T # usol est une matrice de 256x100 éléments.
    ## usol est la solution exacte de l'équation de Burgers. Donc ground truth
    ## des données fournies par le fichier .mat (x et t).
    ## Les lignes correspondent à x et les colonnes à t.

    ############################################################################
    # DATA PROCESSING
    ############################################################################
    # Création des matrices X et T, qui contiennent les coordonnées de chaque point
    #    de la grille. Autrement dit, X contient les coordonnées x de chaque point
    #    de la grille, et T contient les coordonnées t de chaque point de la grille.
    # Note : par "coordonnées" on entend les indices de la grille.
    X, T = np.meshgrid(x,t) # X et T sont des matrices de 100x256 éléments.

    # hstack : empile les matrices horizontalement
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # X_star est une matrice de 25600x2 éléments.
    ## Les lignes correspondent aux coordonnées de chaque point de la grille.
    ## Les colonnes correspondent à x et t de chaque point de la grille.

    # Création du vecteur u_star, qui contient les valeurs de la solution exacte
    #    de l'équation de Burgers aux points de la grille.
    u_star = usol.flatten()[:,None] # u_star est un vecteur de 25600x1 éléments.

    # Limites du domaine
    lb = X_star.min(0) # lb est un vecteur de 2 éléments qui désigne les limites inférieures du domaine.
    ub = X_star.max(0) # ub est un vecteur de 2 éléments qui désigne les limites supérieures du domaine.

    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # xx1 est une matrice de 256x2 éléments. Donc toutes les coordonnées x et t de la première ligne de la grille, soit tout x et t pour t = 0.
    uu1 = usol[0:1,:].T # uu1 est une matrice de 256x1 éléments. Donc toutes les valeurs de la solution exacte pour t = 0.
    xx2 = np.hstack((X[:,0:1], T[:,0:1])) # xx2 est une matrice de 100x2 éléments. Donc toutes les coordonnées x et t de la première colonne de la grille, soit tout x et t pour x = 0.
    uu2 = usol[:,0:1] # uu2 est une matrice de 100x1 éléments. Donc toutes les valeurs de la solution exacte pour x = 0.
    xx3 = np.hstack((X[:,-1:], T[:,-1:])) # xx3 est une matrice de 100x2 éléments. Donc toutes les coordonnées x et t de la dernière colonne de la grille, soit tout x et t pour x = 1.
    uu3 = usol[:,-1:] # uu3 est une matrice de 100x1 éléments. Donc toutes les valeurs de la solution exacte pour x = 1.

    # Plots de vériﬁcation
    if VERIF_PLOTS:
        plt.figure(1)
        plt.plot(xx1[:,0],uu1[:,0],'b.',label='Data (x,t)')
        plt.xlabel('x')
        plt.ylabel('u(x,0)')
        plt.legend()
        # plt.show()

        plt.figure(2)
        plt.plot(xx2[:,1], np.array([0 if i < 0.01 else 1 for i in uu2[:,0]]),'b.',label='Data (x,t)')
        plt.xlabel('t')
        plt.ylabel('u(0,t)')
        plt.legend()
        # plt.show()

        plt.figure(3)
        plt.plot(xx3[:,1], np.array([0 if i < 0.01 else 1 for i in uu3[:,0]]),'b.',label='Data (x,t)')
        plt.xlabel('t')
        plt.ylabel('u(1,t)')
        plt.legend()
        plt.show()


    # Création des matrices X_u_train et u_train, qui contiennent les coordonnées
    #    et les valeurs de la solution exacte aux points de la grille qui serviront
    #    à l'entrainement du réseau de neurones.
    X_u_train = np.vstack([xx1, xx2, xx3]) # X_u_train est une matrice de 456x2 éléments. Donc toutes les coordonnées x et t des points d'entrainement de u.
    X_f_train = lb + (ub-lb)*lhs(2, N_f) # X_f_train est une matrice de 10000x2 éléments. Donc toutes les coordonnées x et t des points d'entrainement de f.
    X_f_train = np.vstack((X_f_train, X_u_train)) # X_f_train est une matrice de 10456x2 éléments. Donc toutes les coordonnées x et t des points d'entrainement de f et de u.
    u_train = np.vstack([uu1, uu2, uu3]) # u_train est une matrice de 456x1 éléments. Donc toutes les valeurs de la solution exacte aux points d'entrainement de u.

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False) # idx est un vecteur de N_u éléments. Donc les indices (au hasard) des points d'entrainement de u.
    X_u_train = X_u_train[idx, :] # X_u_train est une matrice de N_u x 2 éléments. Donc les coordonnées x et t des points d'entrainement de u.
    u_train = u_train[idx,:] # u_train est une matrice de N_u x 1 éléments. Donc les valeurs de la solution exacte aux points d'entrainement de u.
    
    model = PINN(X_u_train, u_train, X_f_train, LAYERS, lb, ub, nu) # Création du réseau de neurones PINN.

    # Entrainement du réseau de neurones PINN.
    model.train(epochs=EPOCHS, learning_rate=LEARING_RATE)

    # Calcul de la solution approchée aux points de la grille.
    u_pred, f_pred = model.predict(X_star)
    print('Error u: %e' % (np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)))
    # On transforme u_pred et f_pred en matrices numpy.
    u_pred = np.array(u_pred)
    f_pred = np.array(f_pred)

    # Prédiction sur tout le domaine.
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic') 
    U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    error = np.abs(U_star-U_pred)

    if PLOTS_3D:
        # Plot de la solution prédite.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, U_pred,
                                linewidth=0, antialiased=False)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_zlabel('$u(x,t)$')
        ax.set_title('Solution prédite')
        ax.view_init(30, 230)
        # plt.show()

        # Plot de la solution exacte.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, U_star,
                                linewidth=0, antialiased=False)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_zlabel('$u(x,t)$')
        ax.set_title('Solution exacte')
        ax.view_init(30, 230)
        # plt.show()

        # Plot de l'erreur de la solution prédite.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, T, error,
                                linewidth=0, antialiased=False)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_zlabel('$u(x,t)$')
        ax.set_title('Erreur de la solution prédite')
        ax.view_init(30, 230)
        plt.show()
    
    # Affichages 2D de la solution exacte, la solution prédite et l'erreur. Avec matplotlib.
    if PLOTS:
        # Création de la figure.
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Plot de la solution exacte.
        ax[0].contourf(T, X, U_star, 100, cmap='jet')
        # Label des axes.
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        # Titre du plot.
        ax[0].set_title('Solution exacte')

        # Plot de la solution prédite.
        ax[1].contourf(T, X, U_pred, 100, cmap='jet')
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].set_title('Solution prédite')

        # Plot de l'erreur de la solution prédite.
        ax[2].contourf(T, X, error, 100, cmap='jet')
        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$x$')
        ax[2].set_title('Erreur de la solution prédite')
        plt.show()