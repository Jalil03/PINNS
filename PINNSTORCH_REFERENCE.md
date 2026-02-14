# Référence PINNsTorch / pinnstorch (notes de projet)

Ce document rassemble, au même endroit, les explications qu’on a construites ensemble sur **ton workspace** `pinn_project` et sur la librairie **`pinnstorch`** (PINNsTorch) : structure, exécution, pipeline d’entraînement, et configuration Hydra.

---

## 0) Ton projet : ce que contient `pinn_project`

Racine : `c:\Users\JL\OneDrive\Desktop\pinnstorch\pinn_project`

- `.venv/` : environnement Python (dépendances installées, dont `pinnstorch`).
- `train.py` : script principal d’entraînement (Navier–Stokes 2D) basé sur des données `.mat`.
- `configs/config.yaml` : ta configuration Hydra principale (mesh/samplers/net/trainer/etc.).
- `data/` : données locales. Tu as `cylinder_nektar_wake.mat` (utilisé par `train.py`).
- `outputs/` : dossiers de runs Hydra (logs + `.hydra/config.yaml`).
- `examples/` : surtout des **artefacts de runs** (logs, configs Hydra résolues, figures).
- `part02/` : second “chapitre”/essai (`train02.py` + `configs/` + `outputs/`) orienté point cloud synthétique.
- `Guide_Module_Data.md` : guide (FR) sur le module `pinnstorch.data` (domaines/mesh/samplers/datamodule).

### Fichiers clés (rôle)

#### `train.py` (ton script)
Tu fournis 3 fonctions “plugin” au framework :

- `read_data_fn(root_path) -> pinnstorch.data.PointCloudData`
  - Charge `data/cylinder_nektar_wake.mat`.
  - Construit les tenseurs `(x, y, t)` et les solutions exactes `{u,v,p}` (ou sous-ensemble).
- `output_fn(outputs, x, y, t) -> outputs`
  - Transforme les sorties du réseau en variables physiques.
  - Exemple Navier–Stokes incompressible : le réseau prédit `psi` et `p`, puis
    - `u = dpsi/dy`
    - `v = -dpsi/dx`
- `pde_fn(outputs, x, y, t, extra_variables) -> outputs`
  - Calcule les résidus PDE (ex: `f_u`, `f_v`) via autograd.
  - `extra_variables` sert typiquement aux problèmes inverses (coefficients à apprendre).

Ensuite tu appelles :

```python
pinnstorch.train(cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=output_fn)
```

#### `configs/config.yaml` (ta config Hydra)
Elle définit :
- `mesh` : comment générer/charger les points (souvent `PointCloud`).
- `train_datasets` : comment échantillonner les points et composer la loss (souvent `MeshSampler`).
- `net` : architecture du réseau (`FCN`, layers, `output_names`).
- `model` : options de `PINNModule` (loss, extra variables, compile/amp/lazy, optimiseur…).
- `trainer` : paramètres Lightning (accélérateur, epochs, devices…).
- `plotting` : fonction de plotting (optionnel).

**Note Hydra importante** : les runs écrivent un dossier `outputs/.../.hydra/config.yaml` qui contient la **config finale résolue** (ce qui a réellement été utilisé).

---

## 1) Comment exécuter ton projet (commandes)

Dans PowerShell :

```powershell
cd "c:\Users\JL\OneDrive\Desktop\pinnstorch\pinn_project"

# activer le venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

# définir PROJECT_ROOT (attendu par pinnstorch/conf/paths)
$env:PROJECT_ROOT = (Get-Location).Path
```

### Run “smoke test” (CPU, court)

```powershell
python .\train.py trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=5 n_train=1000
```

### Run “par défaut”

```powershell
python .\train.py
```

Où regarder :
- logs : dans le dossier du run (selon ta config `hydra.run.dir`)
- config exacte : `...\ .hydra\config.yaml`

> Si tu vois dans les logs “GPU is not found. Using CPU.” c’est normal si CUDA n’est pas dispo : `pinnstorch.utils.set_mode()` force `trainer.accelerator=cpu` et `devices=1`.

---

## 2) Où est la librairie `pinnstorch` sur ta machine

Ton installation est dans `.venv` :

```powershell
python -c "import os, pinnstorch; print(os.path.dirname(pinnstorch.__file__))"
```

Dans ton cas, on a vu :
`...\pinn_project\.venv\Lib\site-packages\pinnstorch\`

Structure principale :

- `pinnstorch/train.py` : l’orchestrateur (Hydra + Lightning).
- `pinnstorch/conf/` : configs Hydra “factory” (defaults, groups).
- `pinnstorch/data/` : mesh/pointcloud, samplers, dataloaders, datamodule.
- `pinnstorch/models/` : `PINNModule`, nets (`FCN`, etc.).
- `pinnstorch/utils/` : gradient utils, logging, instantiation, plotting, etc.

---

## 3) Le pipeline “cœur” (ce qui se passe quand tu lances un train)

### 3.1 `pinnstorch.train(...)` (orchestration)

Fichier : `pinnstorch/train.py`

Idée : `train()` ne connaît pas ta PDE ni tes données : il **instancie** les briques depuis la config Hydra et branche tes fonctions.

Ordre typique :
1) `utils.set_mode(cfg)` : force CPU/GPU + options (amp/cudagraph/etc. selon dispo).
2) Optionnel : instantiate `time_domain` et `spatial_domain` si présents dans la config.
3) Instantiate `mesh` :
   - `Mesh` si tu as un domaine (Interval/Rectangle/…) + TimeDomain,
   - `PointCloud` si tu fournis des points arbitraires (réels/synthétiques),
   - `read_data_fn` est injectée dans le mesh.
4) Instantiate `train_datasets` (souvent des samplers) puis `... (mesh=mesh)`.
5) Instantiate `PINNDataModule` (Lightning) avec les datasets.
6) Instantiate `net` (souvent `FCN`) en lui donnant `lb/ub` du mesh (normalisation).
7) Instantiate `model` (`PINNModule`) en lui donnant `net`, `pde_fn`, `output_fn`.
8) Instantiate callbacks/loggers, instantiate `Trainer`, puis `trainer.fit(...)`.

### 3.2 Point crucial : Hydra instancie à partir de `_target_`

Dans YAML :

```yaml
net:
  _target_: pinnstorch.models.FCN
  _partial_: true
```

Puis dans le code, on verra souvent :
- `hydra.utils.instantiate(cfg.net)(lb=..., ub=...)` si `_partial_: true`

---

## 4) `pinnstorch/models/` : où la loss est calculée (via les samplers)

### 4.1 `PINNModule` (LightningModule)

Fichier : `pinnstorch/models/pinn_module.py`

Rôle :
- appelle `net(...)`
- applique `output_fn` (si fourni)
- délègue la “construction de la loss” aux **samplers**, via un mapping.

Fonctions essentielles :
- `forward(spatial, time)` :
  - `outputs = self.net(spatial, time)`
  - `outputs = output_fn(outputs, *spatial, time)` si `output_fn` existe
- `model_step(batch)` :
  - le `batch` est un dict : `{loss_fn_name: (x, t, u)}`
  - pour chaque entrée, il appelle la fonction `loss_fn` fournie par le sampler
- `training_step(...)` :
  - appelle `model_step()` (ou rejoue un CUDA graph si activé)
  - log `train/loss`
- `validation_step(...)` :
  - calcule `val/loss` + erreurs relatives `val/error_*` sur les variables de solution.

### 4.2 “Qui décide quoi mettre dans la loss ?”

Ce sont les samplers dans `pinnstorch/data/sampler/`.

Autrement dit :
- tu définis dans YAML `solution: [u,v]` et `collection_points: [f_u,f_v]`
- le sampler :
  - appelle `pde_fn` pour produire `f_u`, `f_v`
  - puis additionne les losses sur ces clés + sur `u,v` (si solution fournie)

---

## 5) `pinnstorch/data/` : points, sampling, et datamodule

### 5.1 Mesh vs PointCloud

Fichier : `pinnstorch/data/mesh/mesh.py`

- `Mesh` :
  - construit un maillage régulier depuis un `SpatialDomain` + `TimeDomain`
  - utile si ton domaine est “simple” (Interval, Rectangle, Prism) et discretisé.
- `PointCloud` :
  - construit une grille à partir de points `(x,y,...)` + `t` que tu fournis (réels ou synthétiques)
  - c’est celui que tu utilises actuellement (`read_data_fn` renvoie `PointCloudData`).

Point important :
- Le mesh calcule automatiquement `lb/ub` (min/max sur `(x,y,...,t)`).
- `lb/ub` servent ensuite à normaliser les entrées dans `FCN`.

### 5.2 Samplers : `MeshSampler` (continuous mode)

Fichier : `pinnstorch/data/sampler/mesh_sampler.py`

Cas typiques :
- **avec solution** (`solution: [u,v,p]`) :
  - il sample des couples `(x,t)` et fournit `u(x,t)` comme cible data
- **avec collocation** (`collection_points: [f_u,f_v]`) :
  - il génère aussi des points “sans data” où on force les résidus PDE à 0

Dans `_loss_fn` :
1) `outputs = forward(x, t)`
2) si `collection_points_names` :
   - `outputs = pde_fn(outputs, *x, t, extra_variables)`
3) `loss += loss(outputs[keys=collection_points]) + loss(outputs vs u sur solution)`

### 5.3 `PINNDataModule`

Fichier : `pinnstorch/data/pinn_datamodule.py`

Rôle :
- construit les dataloaders Lightning.
- crée un `function_mapping` qui relie *un identifiant* → *la loss du sampler*.

Important :
- Le train loader peut être “multiple” : plusieurs datasets (conditions) -> plusieurs contributions de loss.

---

## 6) `pinnstorch/conf/` : comprendre Hydra dans pinnstorch

Le dossier `pinnstorch/conf/` est la “factory config” de la lib.

### 6.1 `conf/train.yaml` : la racine et les defaults

Fichier : `pinnstorch/conf/train.yaml`

`defaults:` charge des groupes :
- `data: default`
- `model: default`
- `net: default`
- `callbacks: default`
- `logger: csv`
- `trainer: default`
- `paths: default`
- `extras: default`
- `hydra: default`
- et options `experiment`, `hparams_search`, `debug`, `local`, etc.

**Idée** : ça te donne une config complète “structurée”.

### 6.2 `paths/default.yaml` : `PROJECT_ROOT` et les dossiers

Fichier : `pinnstorch/conf/paths/default.yaml`

- `paths.root_dir: ${oc.env:PROJECT_ROOT}`
- `paths.data_dir: ${paths.root_dir}/data/`
- `paths.output_dir: ${hydra:runtime.output_dir}`

Donc : **tu dois définir** `PROJECT_ROOT` si tu utilises cette config.

### 6.3 `hydra/default.yaml` : où Hydra écrit les runs

Fichier : `pinnstorch/conf/hydra/default.yaml`

Exemple :
- `hydra.run.dir: ${paths.root_dir}/examples/${task_name}/outputs/${now:%H-%M-%S}`

Ça explique pourquoi beaucoup de sorties se retrouvent sous `examples/<task_name>/outputs/...` si tu utilises la conf de la lib.

### 6.4 `_target_` et `_partial_`

Les 2 champs à maîtriser :
- `_target_` : chemin Python à instancier.
- `_partial_: true` : Hydra ne construit pas l’objet final, mais une “factory” qu’on complétera dans le code.

### 6.5 Ton `configs/config.yaml`

Fichier : `configs/config.yaml`

Tu fais :

```yaml
defaults:
  - train
  - _self_
```

Lecture :
1) charger le “train” de pinnstorch (`pinnstorch/conf/train.yaml`)
2) appliquer tes overrides (`_self_`) pour spécialiser mesh/samplers/net/trainer/etc.

**Très utile** : toujours vérifier le résultat dans `outputs/.../.hydra/config.yaml`.

### 6.6 Piège fréquent : erreurs “Key 'lazy' is not in struct”

Ça arrive si tu n’as pas chargé une config `model` complète contenant `lazy`, `amp`, etc.
Solution : baser ta config sur `train` (comme tu fais) ou inclure les groupes nécessaires.

---

## 7) Cheatsheet Hydra (overrides utiles en ligne de commande)

```powershell
# forcer CPU
python .\train.py trainer.accelerator=cpu trainer.devices=1

# réduire la durée
python .\train.py trainer.max_epochs=5 n_train=1000

# changer le dossier de sortie
python .\train.py hydra.run.dir=./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

# désactiver validation/plot
python .\train.py val=false plotting=null
```

---

## 8) Note sur `part02/` (si tu y reviens)

On a observé deux classes d’erreurs dans `part02/outputs/.../train02.log` :

1) `model.lazy` manquant (config `model` incomplète / pas structurée)
2) `model._target_: PINN` introuvable (Hydra n’arrive pas à importer `PINN`)

Pour être compatible `pinnstorch`, le `_target_` du modèle doit typiquement être
`pinnstorch.models.PINNModule` (comme dans `pinnstorch/conf/model/default.yaml`).

---

## 9) “Où modifier quoi” si tu veux étendre la lib

- Nouvelle PDE / nouveaux résidus : tu changes `pde_fn` (ton projet).
- Nouvelles variables physiques dérivées : tu changes `output_fn` (ton projet).
- Changer la façon d’échantillonner / composer la loss :
  - `pinnstorch/data/sampler/*` (ou un sampler custom dans ton projet + `_target_`).
- Changer l’architecture :
  - `pinnstorch/models/net/neural_net.py` (`FCN`) ou un net custom.
- Changer le “training engine” :
  - `pinnstorch/models/pinn_module.py` (Lightning hooks, logs, AMP, compile…).
- Changer les defaults/configs :
  - `pinnstorch/conf/*` (ou overrides dans `configs/config.yaml` + CLI).

