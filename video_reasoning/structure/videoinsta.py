import logging

from video_reasoning.state.clip import ClipState

logger = logging.getLogger("root")


class Clip:
    def __init__(self, state: ClipState = None):
        self.state = state
        logger.info("Initialized clip")


class Relation:
    def __init__(self, source: Clip, target: Clip):
        self.source: Clip = source
        self.target: Clip = target
        logger.info("Initialized relation")


class VideoINSTA:
    def __init__(self, root_clip: Clip):
        self.clips = list[Clip]()
        self.relations = list[Relation]()

        self.root: Clip = root_clip
        self.add_clip(self.root)

        logger.info("Initialized VideoINSTA")

    def get_relations_from_clip(self, clip) -> list[Relation]:
        return [relation for relation in list(self.relations) if relation.source == clip]

    def get_successors_of_clip(self, clip) -> list[Clip]:
        return [relation.target for relation in self.get_relations_from_clip(clip)]

    def get_predecessors_of_clip(self, clip) -> list[Clip]:
        return [relation.source for relation in self.get_relations_from_clip(clip)]

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)

    def add_relations(self, relations: list[Relation]) -> None:
        self.relations.extend(relations)

    def add_clip(self, clip: Clip) -> None:
        self.clips.append(clip)

    def add_clips(self, clips: list[Clip]) -> None:
        self.clips.extend(clips)

    def get_derivable_clips(self) -> list[Clip]:
        return [clip for clip in self.clips if not clip.state.derived and clip != self.root]

    def get_unranked_clips(self) -> list[Clip]:
        return [clip for clip in self.clips if not clip.state.ranking and clip != self.root]

    def get_sub_clips(self) -> list[Clip]:
        return [clip for clip in self.clips if not self.get_successors_of_clip(clip)]

    def get_derived_root_successor_clips(self) -> list[Clip]:
        root_successors = self.get_successors_of_clip(self.root)
        return [clip for clip in root_successors if clip.state.derived]

    def get_decidable_sub_clips(self) -> list[Clip]:
        return [clip for clip in self.clips if
                # consider sub clips
                (not self.get_successors_of_clip(clip)
                 # consider sub clips that are not waiting
                 and not clip.state.waiting)]

    def get_concludable_clips(self) -> list[Clip]:
        return [clip for clip in self.clips if
                # consider sub clips
                (not self.get_successors_of_clip(clip)
                 # consider sub clips that have states that are derived
                 and clip.state.derived)]
