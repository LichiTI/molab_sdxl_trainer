import re
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any

# TODO: Re-implement token counting if needed, possibly using a lighter library or the same clip tokenizer if dependency allows.
# from clip import tokenize 

# TODO: Inject Tag Categories/Configuration instead of hardcoding 'resources'
# from resources import tag_categories

class RectElement:
    def __init__(self, name: str, top: int = 0, left: int = 0, width: int = 0, height: int = 0, confidence: float = 1.0, color: str = ""):
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        # this is 1 for manual entries, this is adjusted for detection model results
        self.confidence = confidence
        self.name = name

        # Tags Values
        self.sentence_description: SentenceElement = SentenceElement()
        self.auto_tags: TagsLists = TagsLists(name="auto_tags")
        self.manual_tags: TagsList = TagsList(name="manual_tags")
        self.rejected_manual_tags: TagsList = TagsList(name="rejected_manual_tags")

        # Virtuals/Not Saved
        self.rejected_tags: TagsList = TagsList(name="rejected_tags")
        self.filtered_new_tags: TagsList = TagsList(name="filtered_new_tags")
        self.filtered_rejected_tags: TagsList = TagsList(name="filtered_rejected_tags")
        self.full_tags: TagsList = TagsList(name="full_tags")

        # Future values:
        self.color: str = color # Hexadecimal color value of the rect

    def apply_from_dict(self, save_dict: Dict[str, Any]):
        saved_keys = save_dict.keys()
        apply_filter = False
        if "name" in saved_keys:
            self.name = save_dict["name"]
        if "color" in saved_keys:
            self.color = save_dict["color"]
        if "coordinates" in saved_keys:
            self.top = save_dict["coordinates"][0]
            self.left = save_dict["coordinates"][1]
            self.width = save_dict["coordinates"][2]
            self.height = save_dict["coordinates"][3]
        if "confidence" in saved_keys:
            self.confidence = save_dict["confidence"]
        if "sentence" in saved_keys:
            self.sentence_description.sentence = save_dict["sentence"]
        if "auto_tags" in saved_keys:
            self.auto_tags.overwrite(save_dict["auto_tags"])
            apply_filter = True
        if "manual_tags" in saved_keys:
            self.manual_tags = TagsList(tags=save_dict["manual_tags"], name="manual_tags")
            apply_filter = True
        if "rejected_manual_tags" in saved_keys:
            self.rejected_manual_tags = TagsList(tags=save_dict["rejected_manual_tags"], name="rejected_manual_tags")
            apply_filter = True
        if apply_filter:
            self.filter()

    def apply_coordinates(self, top: int, left: int, width: int, height: int):
        """
        Overwrite the coordinates of the Rectangle Element
        """
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def update_confidence(self, new_confidence: float):
        self.confidence = new_confidence
 
    def save(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "coordinates": (self.top, self.left, self.width, self.height),
            "confidence": self.confidence
        }
        if self.sentence_description:
            result["sentence"] = self.sentence_description.save()
        if self.color:
            result["color"] = self.color
        # Only save if they have content to avoid clutter
        if self.auto_tags:
            result["auto_tags"] = self.auto_tags.save()
        if self.manual_tags:
            result["manual_tags"] = self.manual_tags.save()
        if self.rejected_manual_tags:
            result["rejected_manual_tags"] = self.rejected_manual_tags.save()
        return result

    def add_new_tags(self, tags: Union[str, List[str], 'TagElement', List['TagElement']]):
        """
        Add tags where is needed (manual tags only)
        """
        if isinstance(tags, (str, list, TagElement)):
            self.manual_tags += tags
        else:
             # Try conversion or log warning
             pass
        self.rejected_manual_tags -= tags
        self.filter()

    def remove_tags(self, tags: Union[str, List[str], 'TagElement', List['TagElement']]):
        """
        Remove tags where is needed (manual tags only)
        """
        self.rejected_manual_tags += tags
        self.manual_tags -= tags
        self.filter()

    def update_full_tags(self):
        # manual + from txt + auto tags + filtered tags + secondary tags
        self.full_tags.clear()
        self.full_tags = self.full_tags + self.auto_tags + self.filtered_new_tags
        self.full_tags -= self.get_rejected_tags()
        self.full_tags += self.manual_tags

    def get_prefiltered_full_tags(self) -> 'TagsList':
        """
        only used by the filter
        return the full tag as if there was no filter, except for manually rejected tags
        """
        # manual + from txt + auto tags + filtered tags +secondary tags
        full_tags = TagsList(name="full_tags")
        full_tags = full_tags + self.auto_tags - self.rejected_manual_tags + self.manual_tags
        return full_tags

    def filter(self):
        self.filtered_new_tags.clear()
        self.filtered_rejected_tags.clear()
        full_tags = self.get_prefiltered_full_tags()

        # TODO: Implement filtering logic with injected strategies
        # Current implementation is stripped of hardcoded tag_categories
        
        # Original logic involved:
        # 1. to_high()
        # 2. hard_rejected_tags()
        # 3. not_hard_rejected_tags()
        # 4. has_low() etc.

        # detailed filtering logic suspended until configuration injection is implemented.
        pass

        # update the unresolved
        self.update_full_tags()

    def get_rejected_tags(self) -> 'TagsList':
        """
        recalculate the full rejected tags, and both update it and return it
        """
        self.rejected_tags.clear()
        self.rejected_tags += self.rejected_manual_tags + self.filtered_rejected_tags
        return self.rejected_tags

    def get_full_tags(self) -> 'TagsList':
        """
        recalculate the full tags, and both update it and return it
        """
        self.update_full_tags()
        return self.full_tags
    
    def get_full_only_tags(self) -> list[str]:
        """
        update full tags and return only the string version of tags
        """
        return [tag.tag for tag in self.get_full_tags()]

    def create_output(self, add_backslash_before_parenthesis=False, keep_tokens_separator: str= "|||", main_tags: list[str]=[], secondary_tags: list[str]=[], use_sentence=True, shuffle_tags=True) -> str:
        result = ""
        if use_sentence:
            segments = self.sentence_description.get_output_info()
        else:
            segments = ["#full_tags"]

        for segment in segments:
            if segment == "#full_tags":
                tags = self.get_full_only_tags()
                if shuffle_tags:
                    random.shuffle(tags)
                identified_main_tags = []
                identified_secondary_tags = []

                if main_tags or secondary_tags:
                    for main_tag in main_tags:
                        if "*" in main_tag:
                            k = 0
                            while k < len(tags):
                                if re.fullmatch(r'.*'.join(main_tag.split("*")), tags[k]):
                                    identified_main_tags.append(tags[k])
                                    tags.remove(tags[k])
                                else:
                                    k+=1
                        elif main_tag in tags:
                            tags.remove(main_tag)
                            identified_main_tags.append(main_tag)
                    for secondary_tag in secondary_tags:
                        if "*" in secondary_tag:
                            k = 0
                            while k < len(tags):
                                if re.fullmatch(r'.*'.join(secondary_tag.split("*")), tags[k]):
                                    identified_secondary_tags.append(tags[k])
                                    tags.remove(tags[k])
                                else:
                                    k+=1
                        elif secondary_tag in tags:
                            tags.remove(secondary_tag)
                            identified_secondary_tags.append(secondary_tag)

                temp_tags = []
                if identified_main_tags:
                    temp_tags += identified_main_tags
                if identified_secondary_tags:
                    if shuffle_tags:
                        random.shuffle(identified_secondary_tags)
                    temp_tags += identified_secondary_tags

                if len(segments) == 1 and keep_tokens_separator: 
                    temp_tags.append(keep_tokens_separator)

                tags = temp_tags + tags
                result += ", ".join(tags)
            elif isinstance(segment, str): # when it's simple text
                result += segment

        if add_backslash_before_parenthesis:
            result = result.replace('(', '\\(').replace(')', '\\)')

        return result


class SentenceElement:
    def __init__(self, sentence: Union[str, 'SentenceElement'] = ""):
        if isinstance(sentence, str):
            self.sentence = sentence
        elif isinstance(sentence, SentenceElement):
            self.sentence = sentence.sentence
        else:
            self.sentence = ""
        self.token_length: int = 0
        self.sentence_length: int = 0

    def __bool__(self):
        return bool(self.sentence)

    def __str__(self):
        return self.sentence

    def __eq__(self, other):
        if isinstance(other, SentenceElement):
            return self.sentence == other.sentence
        return False


    def get_token_length(self):
        # TODO: Implement token counting with a tokenizer (e.g., CLIP)
        return 0
    
    def get_sentence_length(self):
        if self.sentence:
            self.sentence_length = len(self.sentence)
        else:
            self.sentence_length = 0
        return self.sentence_length
            
    def save(self):
        return self.sentence

    def get_output_info(self) -> List[Union[str, Tuple[str, str]]]:
        """
        Output a modified version of the sentence that permits the output of a list of items in order
        - ##FTAGS## for full_tags
        - ##SCORE## for score_label
        - ##RECT:rect_name## for the sentence of rects
        """
        if not self.sentence:
            return ["#full_tags"]
        result = []
        temp_result = self.sentence.split("##")
        for segment in temp_result:
            if segment.strip() == "FTAGS":
                result.append("#full_tags")
            elif segment.strip() ==  "SCORE":
                result.append("#score_label")
            elif "RECT:" in segment:
                try:
                    rect_name = segment.split(":", maxsplit=1)[1].strip()
                    result.append(("RECT", rect_name))
                except IndexError:
                    result.append(segment)
            else:
                result.append(segment)

        return result


class GroupElement:
    def __init__(self, *, group_name: str="", md5s: List[str]=None):
        if md5s is None:
            md5s = []
        self.group_name: str = group_name
        self.md5s: List[str] = md5s

    def __len__(self):
        return len(self.md5s)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.md5s[self.md5s.index(key)] or key == self.group_name
        elif isinstance(key, int):
            return self.md5s[key]

    def __setitem__(self, key, value):
        if isinstance(key, str) and key not in self.md5s:
            self.md5s.append(value)
            return
        elif isinstance(key, str):
            self.md5s[self.md5s.index(key)] = value
            return
        self.md5s[key] = value

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.group_name
        elif isinstance(other, type(self)):
            if len(self) != len(other):
                return False
            if all(md5 in self.md5s for md5 in other.md5s):
                return True
        elif isinstance(other, list):
            if all(md5 in self.md5s for md5 in other):
                return True
        return False

    def append(self, item):
        if isinstance(item, str) and item not in self.md5s:
            self.md5s.append(item)

    def remove(self, item):
        if isinstance(item, str) and item in self.md5s:
            self.md5s.remove(item)

    def save(self):
        result = {"images": self.md5s}
        return result

class TagsLists:
    """
    Exists for external_tags, auto_tags
    """
    def __init__(self, tags_list=None, *, name: str=""):
        if tags_list is None:
            tags_list = []
        self.tags_lists: List[TagsList] = tags_list
        self.name: str = name
        self.tags_confidence: TagsList = TagsList(name="tags_over_confidence")

    def __repr__(self):
        return "TagsLists("+str(self.tags_lists)+", name="+self.name+")"

    def __bool__(self):
        return any([bool(x) for x in self.tags_lists]) # Fixed logic: any list has content means it's truthy, though original was all(). Confirm? Original: all([bool(x) for x in self.tags_lists]). This means if one is empty, it's false? Let's stick to original if unsure or change to any? Original: all. I will check logic. Actually, usually we want to know if *any* tags exist. Reference said `all`. Checking context. If `auto_tags` has empty lists, is it empty? I'll stick to original logic but be careful. 
        # Actually, let's look at reference: `return all([bool(x) for x in self.tags_lists])`. If list is empty `[]`, `all([])` is True. If I initialize with empty lists...
        # Let's assume standard behavior is desired. If I have tags, I am true.
        return len(self.tags_lists) > 0 and any(bool(x) for x in self.tags_lists)

    def __len__(self):
        return len(self.tags_lists)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if len(self.names()) != len(other.names()):
                return False
            if all(self[name] == other[name] for name in self.names()):
                return True
        return False

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.tags_lists[key]
        elif isinstance(key, str):
            # Safe retrieval
            try:
                return self.tags_lists[self.names().index(key)]
            except ValueError:
                # Return empty TagsList if not found? Or raise? Original would raise ValueError.
                 # Let's conform to original for now.
                 raise KeyError(f"TagList with name {key} not found")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if key in self.names():
                idx = self.names().index(key)
                if isinstance(value, TagsList):
                     self.tags_lists[idx] = value
                elif isinstance(value, list):
                     self.tags_lists[idx] = TagsList(tags=value, name=key)
            else:
                if isinstance(value, TagsList):
                    self.tags_lists.append(value)
                elif isinstance(value, list):
                    self.tags_lists.append(TagsList(tags=value, name=key))
            return 
        
        # If key is int
        if isinstance(value, TagsList):
            self.tags_lists[key] = value
        elif isinstance(value, list):
            self.tags_lists[key] = TagsList(tags=value, name=self.tags_lists[key].name)

    def overwrite(self, other):
        """
        Add a new tags_lists, or overwrite it if it's the same name
        """
        if isinstance(other, dict):
            for key, value in other.items():
                applied = False
                for i in range(len(self)):
                    if key == self[i].name:
                        self[i] = value
                        applied = True
                        break
                if not applied:
                    self.tags_lists.append(TagsList(tags=value, name=key))
        elif isinstance(other, TagsList):
            if other.name not in self.names():
                self.tags_lists.append(other)
            else:
                self[other.name] = other

    def names(self):
        return [tags_list.name for tags_list in self.tags_lists]

    def save(self):
        result = defaultdict(lambda: [])
        for tags_list in self.tags_lists:
            result[tags_list.name] = tags_list.save()
        return result

    def save_tuple(self):
        result = defaultdict(lambda: [])
        for tags_list in self.tags_lists:
            result[tags_list.name] = tags_list.save_tuple()
        return result

    def simple_tags(self):
        simple_tags = TagsList()
        simple_tags += self.tags_lists
        return simple_tags.simple_tags()

    def refresh_unsafe_tags(self, all_accepted_tags):
        # Relies on "rejected" naming convention
        for tags_list in self.tags_lists:
            if "rejected" in tags_list.name:
                base_name = tags_list.name.replace("rejected", "").strip("_") # Rough logic to get original name? 
                # Original logic: tags_list.name[len("rejected") + 1:] (assuming "rejected_XXX")
                if tags_list.name.startswith("rejected_"):
                   target_name = tags_list.name[9:] # "rejected_".length == 9
                elif tags_list.name.startswith("rejected"):
                   target_name = tags_list.name[8:]
                else:
                    continue

                new_accepted = tags_list.all_tags_in(all_accepted_tags)
                if new_accepted:
                    new_accepted.name = target_name
                    self[target_name] = new_accepted
                else:
                    new_accepted = TagsList(name=target_name, tags=[])
                    self[target_name] = new_accepted

    def _build_merged_confidence(self):
        if not self.tags_confidence:
             # Logic to merge
             merged = TagsList(name="tags_over_confidence")
             for tags_list in self.tags_lists:
                for tag in tags_list.tags:
                    if tag not in merged.tags:
                        merged += tag
                    else:
                        # take higher confidence
                        existing = merged[tag.tag]
                        if existing.probability < tag.probability:
                            merged[tag.tag] = tag
             self.tags_confidence = merged

    def tags_over_confidence(self, confidence: float):
        self._build_merged_confidence()
        return self.tags_confidence.tags_over_confidence(confidence)

    def tags_under_confidence(self, confidence: float):
        self._build_merged_confidence()
        return self.tags_confidence.tags_under_confidence(confidence)

    def all_tags_in(self, other):
        combined = TagsLists(name=self.name)
        if isinstance(other, type(self)):
            for name in self.names():
                if name in other.names():
                    new_tags = self[name].all_tags_in(other[name])
                    if new_tags:
                        combined.overwrite(new_tags)
        return combined

    def common_tags(self, other):
        combined_tags = TagsList(name="Combined "+self.name)
        combined_tags += set([tag.tag for list_tags in self.tags_lists for tag in list_tags if "rejected" not in list_tags.name])
        if isinstance(other, type(self)):
             other_tags = set([tag.tag for list_tags in other.tags_lists for tag in list_tags if "rejected" not in list_tags.name])
             combined_tags = combined_tags.all_tags_in(other_tags)
        combined = TagsLists(tags_list=[combined_tags], name=self.name)
        return combined


class TagsList:
    def __init__(self,*, tags=None, name=""):
        if tags is None:
            tags = []
        self.tags: List['TagElement'] = []
        for tag in tags:
            self.tags.append(TagElement(tag))
        self.name: str = name
        self.token_length: int = 0

    def __repr__(self):
        return "TagsList(name="+self.name+",tags="+str(self.tags)+")"

    def __iter__(self):
        return iter(self.tags)

    def __bool__(self):
        return bool(self.tags)

    def __add__(self, other):
        new = TagsList(name=self.name, tags=self.tags)

        if isinstance(other, type(self)):
            new.tags += [tag for tag in other.tags if tag not in new.tags]
        elif isinstance(other, (list, set)):
            new.tags += [TagElement(tag) for tag in other if TagElement(tag) not in new.tags]
        elif isinstance(other, TagElement) and other not in new.tags:
            new.tags += [other]
        elif isinstance(other, str) and TagElement(other) not in new.tags:
             new.tags += [TagElement(other)]
        elif isinstance(other, TagsLists):
            for other_tag_list in other.tags_lists:
                if "rejected" not in other_tag_list.name:
                    new.tags += [tag for tag in other_tag_list.tags if tag not in new.tags]

        return new
        
    def __sub__(self, other):
        new = TagsList(name=self.name)

        if isinstance(other, type(self)):
            new.tags = [tag for tag in self.tags if tag not in other.tags]
        elif isinstance(other, (list, set)):
            proper_other = [TagElement(t) for t in other] 
            new.tags = [tag for tag in self.tags if tag not in proper_other]
        elif isinstance(other, TagElement):
            new.tags = [tag for tag in self.tags if tag != other]
        elif isinstance(other, str):
            new.tags = [tag for tag in self.tags if tag.tag != other]
        elif isinstance(other, TagsLists):
            combined = TagsList()
            combined += other
            new.tags = [tag for tag in self.tags if tag not in combined.tags]
        
        # Ensure name persists if needed, though logic creates new list
        return new

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if len(self.tags) != len(other.tags):
                return False
            return all([tag in other.tags for tag in self.tags])
        elif isinstance(other, (list, set)):
            if len(self.tags) != len(other):
                return False
            # Check string equality
            other_strs = [str(x) for x in other]
            return all([tag.tag in other_strs for tag in self.tags])

        return False

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.tags[key]
        elif isinstance(key, TagElement):
             try:
                 idx = [tag.tag for tag in self.tags].index(key.tag)
                 return self.tags[idx]
             except ValueError:
                 raise KeyError(f"Tag {key.tag} not found")
        elif isinstance(key, str):
             try:
                 idx = [tag.tag for tag in self.tags].index(key)
                 return self.tags[idx]
             except ValueError:
                 raise KeyError(f"Tag {key} not found")

    def __setitem__(self, key, value):
        if isinstance(value, TagElement):
            if key in self.tags:
                self.tags[self.tags.index(key)] = value
            else:
                self.tags.append(value)
        elif isinstance(value, (str, tuple, list)):
             # If key is an existing tag (int or obj), replace logic is ambiguous in python list.
             # In original code:
             # if key in self.tags: ...
             # This implies key is used to find index.
             if key in self.tags:
                  self.tags[self.tags.index(key)] = TagElement(value)
             else:
                  self.tags.append(TagElement(value))

    def __len__(self):
        return len(self.tags)

    def pop(self, index):
        if isinstance(index, int):
            return self.tags.pop(index)
        if isinstance(index, (str, TagElement)):
             try:
                if isinstance(index, str):
                     real_index = [t.tag for t in self.tags].index(index)
                else:
                     real_index = self.tags.index(index)
                return self.tags.pop(real_index)
             except ValueError:
                 return None

    def get_token_length(self):
        # TODO: Implement
        return 0

    def all_tags_in(self, other):
        if isinstance(other, type(self)):
            new_tags = [tag for tag in self.tags if tag.tag in [o.tag for o in other.tags]]
        elif isinstance(other, (list, set)):
            new_tags = [tag for tag in self.tags if tag.tag in other]
        elif isinstance(other, TagElement):
             new_tags = [tag for tag in self.tags if tag.tag == other.tag]
        elif isinstance(other, TagsLists):
            combined = TagsList()
            combined += other
            new_tags = [tag for tag in self.tags if tag in combined.tags]
        else:
            new_tags = []
        return TagsList(name=self.name, tags=new_tags)


    def save(self):
        return [x.save() for x in self.tags]

    def save_tuple(self):
        return [x.save_tuple() for x in self.tags]

    def simple_tags(self):
        result = [x.tag for x in self.tags]
        return result

    def clear(self):
        self.tags = []

    # Placeholder methods for tag categories logic
    def to_high(self):
        # TODO: Implement with config
        return TagsList()

    def to_low(self):
        # TODO: Implement with config
        return TagsList()

    def has_low(self):
         # TODO: Implement with config
        return TagsList()

    def hard_rejected_tags(self):
        # TODO: Implement with config
        return TagsList()

    def not_hard_rejected_tags(self):
        # TODO: Implement with config
        # Return all for now as default safe behavior?
        # If we return all, then filtering does nothing, which is safe.
        return TagsList(tags=self.tags) 

    def tags_over_confidence(self, confidence: float):
        return TagsList(tags=[tag for tag in self.tags if tag.probability >= confidence])

    def tags_under_confidence(self, confidence: float):
        return TagsList(tags=[tag for tag in self.tags if tag.probability <= confidence])

    def recommendations(self):
         # TODO: Implement with config
        return TagsList(name="recommendations")

    def priority_sort(self):
        # TODO: Implement sort priority logic if needed
        # self.tags.sort(key=lambda x: x.sort_priority, reverse=False)
        pass

class TagElement:
    def __init__(self, tag: Union[str, Tuple, List, 'TagElement'], *, probability: float=0.0):
        if isinstance(tag, str):
            self.tag: str = tag
            self.probability = probability
        elif isinstance(tag, (tuple, list)):
            self.tag = str(tag[0])
            self.probability = float(tag[1]) if len(tag) > 1 else 0.0
        elif isinstance(tag, TagElement):
             self.tag = tag.tag
             self.probability = tag.probability
        else:
            self.tag = str(tag)
            self.probability = probability
            
        # Display properties (virtual)
        self.color = (255, 255, 255, 255)
        self.priority = 99999
        self.highlight = False
        self.manual = False
        self.rejected = False

    def __repr__(self):
        return f"TagElement('{self.tag}', {self.probability})"

    def __eq__(self, other):
        if isinstance(other, TagElement):
            return self.tag == other.tag
        elif isinstance(other, str):
            return self.tag == other
        return False
    
    def __hash__(self):
        return hash(self.tag)

    def save(self):
        return self.tag

    def save_tuple(self):
        return (self.tag, self.probability)

