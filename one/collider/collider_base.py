import numpy as np
import one.scene.scene as oss


class ColliderBase:
    def __init__(self):
        self.scene = oss.Scene()
        self._actors = ()
        self._actor_qs_slice = {}
        self._compiled = False
        self._check_pairs = None
        self._pair_items = None

    def append(self, entity):
        """Add robot or obstacle to scene"""
        self.scene.add(entity)

    def compile(self):
        if not self._actors:
            raise RuntimeError('actors must be set before compile')
        pair_items = []
        item_index = {}
        check_pairs = []

        def get_idx(obj):
            idx = item_index.get(obj)
            if idx is None:
                idx = len(pair_items)
                pair_items.append(obj)
                item_index[obj] = idx
            return idx

        # 1) Self-collision
        for actor in self._actors:
            compiled = actor.structure.compiled
            n_lnks = len(actor.runtime_lnks)
            for i in range(n_lnks):
                for j in range(i + 1, n_lnks):
                    lnk = actor.runtime_lnks[i]
                    lnk2 = actor.runtime_lnks[j]
                    if not lnk.collisions or not lnk2.collisions:
                        continue
                    if not self._should_collide(lnk, lnk2):
                        continue
                    pair = (min(i, j), max(i, j))
                    if pair in compiled.collision_ignores_idx:
                        continue
                    idx_a = get_idx(lnk)
                    idx_b = get_idx(lnk2)
                    check_pairs.append([idx_a, idx_b])
        # 2) Actor vs SceneObject
        for actor in self._actors:
            lnks = actor.runtime_lnks
            valids = [(i, lnk)
                      for i, lnk in enumerate(lnks)
                      if lnk.collisions]
            for sobj in self.scene.sobjs:
                if not sobj.collisions:
                    continue
                for lidx, lnk in valids:
                    if not self._should_collide(lnk, sobj):
                        continue
                    idx_a = get_idx(lnk)
                    idx_b = get_idx(sobj)
                    check_pairs.append([idx_a, idx_b])
        # 3) Actor vs Non-Actor Robot
        non_actors = [mb for mb in self.scene.mecbas
                      if mb not in self._actors]
        for actor in self._actors:
            valids_a = [(i, lnk)
                        for i, lnk in enumerate(actor.runtime_lnks)
                        if lnk.collisions]
            for non_actor in non_actors:
                valids_b = [(i, lnk)
                            for i, lnk in enumerate(non_actor.runtime_lnks)
                            if lnk.collisions]
                for actor_lidx, actor_lnk in valids_a:
                    for non_actor_lidx, non_actor_lnk in valids_b:
                        if not self._should_collide(actor_lnk, non_actor_lnk):
                            continue
                        idx_a = get_idx(actor_lnk)
                        idx_b = get_idx(non_actor_lnk)
                        check_pairs.append([idx_a, idx_b])
        # 4) Actor vs Actor
        n_actors = len(self._actors)
        for i in range(n_actors):
            actor_a = self._actors[i]
            valids_a = [(idx, lnk)
                        for idx, lnk in enumerate(actor_a.runtime_lnks)
                        if lnk.collisions]
            for j in range(i + 1, n_actors):
                actor_b = self._actors[j]
                valids_b = [(idx, lnk)
                            for idx, lnk in enumerate(actor_b.runtime_lnks)
                            if lnk.collisions]
                for lidx_a, lnk_a in valids_a:
                    for lidx_b, lnk_b in valids_b:
                        if not self._should_collide(lnk_a, lnk_b):
                            continue
                        idx_a = get_idx(lnk_a)
                        idx_b = get_idx(lnk_b)
                        check_pairs.append([idx_a, idx_b])
        self._pair_items = pair_items
        self._check_pairs = np.array(check_pairs, dtype=np.int32)
        self._compiled = True
        self._post_compile()

    @property
    def actors(self):
        return self._actors

    @actors.setter
    def actors(self, actors):
        if not actors:
            raise ValueError('actors cannot be empty')
        self._actors = tuple(actors)
        self._rebuild_mapping()

    def _post_compile(self):
        """Subclass hook"""
        pass

    def is_collided(self, qs):
        raise NotImplementedError

    def _rebuild_mapping(self):
        self._actor_qs_slice.clear()
        offset = 0
        for actor in self._actors:
            if actor not in self.scene.mecbas:
                raise RuntimeError('All actors must be added to the scene')
            ndof = actor.ndof
            self._actor_qs_slice[actor] = slice(offset, offset + ndof)
            offset += ndof

    def _should_collide(self, a, b):
        ga = a.collision_group
        gb = b.collision_group
        aa = a.collision_affinity
        ab = b.collision_affinity
        return bool((aa & gb) and (ab & ga))
