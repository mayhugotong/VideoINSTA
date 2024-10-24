import json
import logging
import re

logger = logging.getLogger("root")


def parse_list(text: str):
    """
    Parses a list from a given text.

    Parameters
    ----------
    text: str
        The text to parse the list from.

    Returns
    -------
    list[str]
        The parsed list.
    """
    text = text.strip()

    # splits = re.split("(\d+\. |\d+\) |- |\* |\+ |• |[A-Za-z]\. |[A-Za-z]\) |[iv]+\. |[iv]+\) )", text)
    splits = re.split("(\d+\. |\d+\) |- |\* |\+ |• )", text)

    if len(splits) <= 2:
        return []

    # discard first split (is either empty if the text started with a list identifier or has some noise)
    splits = splits[1:]

    # remove last item if the length of splits is odd (in this case the last item is a single identifier)
    if len(splits) % 2 == 1:
        splits = splits[:-1]

    # remove noise of the last item
    last_item = splits[-1]
    last_item = last_item.split("\n")[0]
    splits[-1] = last_item

    items = []

    # get every list item and discard the list identifiers
    for i in range(1, len(splits), 2):
        item = splits[i].strip().lower()
        # discard empty items and items with more than 3 words
        # if item != "" and len(item.split(" ")) <= 3:
        if item != "":
            items.append(item)

    return items


def filter_list_of_objects(objects: list[str]):
    # remove everything after a starting bracket
    objects = [o.split("(")[0].strip() for o in objects]
    objects = [o.split("[")[0].strip() for o in objects]
    objects = [o.split("{")[0].strip() for o in objects]

    # remove everything after linebreaks
    objects = [o.split("\n")[0].strip() for o in objects]

    # replace commas with whitespace
    objects = [o.replace(", ", " ") for o in objects]
    objects = [o.replace("; ", " ") for o in objects]
    objects = [o.replace(": ", " ") for o in objects]

    # remove unwanted objects
    unwanted_objects = ["action", "actions", "none", "c", "character", "characters", "character c"]
    unwanted_object_parts = ["#c", "#c's", "c (#c)", "#c (c)", "c's"]

    # filter out objects that are unwanted
    objects = [o for o in objects if o.lower() not in unwanted_objects and o not in unwanted_object_parts]

    # remove unwanted object parts from objects
    for unwanted_object_part in unwanted_object_parts:
        objects = [o.replace(unwanted_object_part, "").strip() for o in objects]

    # remove objects that contain more than four words
    objects = [o for o in objects if len(o.split(" ")) <= 4]

    # filter out objects that are unwanted (again if an unwanted object was created through the steps before)
    objects = [o for o in objects if o.lower() not in unwanted_objects and o not in unwanted_object_parts]

    # filter out duplicates
    objects = list(set(objects))

    # sort the list of objects to ensure reproducibility through deterministic behavior
    objects.sort()

    return objects


def parse_option_answer_from_text(text: str) -> list[str]:
    # strip the text and convert it to lowercase
    text = text.strip().lower().replace("_", " ")

    # define the possible options
    numbers = ["0", "1", "2", "3", "4"]
    letters = ["a", "b", "c", "d", "e"]
    numbers_options = ["option 0", "option 1", "option 2", "option 3", "option 4"]
    letters_options = ["option a", "option b", "option c", "option d", "option e"]

    # in the following, try to find the best parsing step by step
    answer = ""

    # give most priority to the answer if it is explicitly marked
    if answer == "" and "answer:" in text:
        # find the answer in the completion marked by the word "answer:"
        answer = text.split("answer:")[-1].strip()[:8]

        # check if the answer is a valid option and return if so
        if answer in numbers_options:
            return [answer]
        if answer in letters_options:
            return [numbers_options[letters_options.index(answer)]]

        # try again but check if there is some other noise as well after the keyword
        # or if there is just the single identifier
        answer = text.split("answer:")[-1].strip()[:30]
        for number in numbers:
            if number in answer:
                return [f"option {number}"]
        for letter in letters:
            if letter in answer:
                return [f"option {letters.index(letter)}"]

        answer = ""

    logger.debug(f"First version of answer parsing: {answer}")

    if answer == "" and "option" in text:
        # find last option in the completion
        # pattern = r'\boption(?: |_)?\d\b'
        pattern = r'\boption(?: |_)?(?:\d|[a-e])\b'
        all_matches = re.findall(pattern, text)
        answer = all_matches[-1] if all_matches else ""

        if answer in numbers_options:
            return [answer]
        elif answer in letters_options:
            return [numbers_options[letters_options.index(answer)]]
        else:
            answer = ""

    logger.debug(f"Second version of answer parsing: {answer}")

    if answer == "" and "number " in text:
        answer = text.split("number ")[-1].strip()[:8]
        answer = answer if "option" in answer else ""

        if answer == "":
            answer = text.split("number ")[-1].strip()[:1]
            answer = f"option {answer}" if answer in numbers else ""
            answer = f"option {letters.index(answer)}" if answer in letters else ""

        if answer in numbers_options:
            return [answer]
        elif answer in letters_options:
            return [numbers_options[letters_options.index(answer)]]
        else:
            answer = ""

    logger.debug(f"Third version of answer parsing: {answer}")

    if answer == "" and "index " in text:
        answer = text.split("index ")[-1].strip()[:8]
        answer = answer if "option" in answer else ""

        if answer == "":
            answer = text.split("index ")[-1].strip()[:1]
            answer = f"option {answer}" if answer in numbers else ""
            answer = f"option {letters.index(answer)}" if answer in letters else ""

        if answer in numbers_options:
            return [answer]
        elif answer in letters_options:
            return [numbers_options[letters_options.index(answer)]]
        else:
            answer = ""

    logger.debug(f"Fourth version of answer parsing: {answer}")

    if answer == "" and "id " in text:
        answer = text.split("id ")[-1].strip()[:8]
        answer = answer if "option" in answer else ""

        if answer == "":
            answer = text.split("id ")[-1].strip()[:1]
            answer = f"option {answer}" if answer in numbers else ""
            answer = f"option {letters.index(answer)}" if answer in letters else ""

        if answer in numbers_options:
            return [answer]
        elif answer in letters_options:
            return [numbers_options[letters_options.index(answer)]]
        else:
            answer = ""

    logger.debug(f"Fifth version of answer parsing: {answer}")

    if answer == "" and "id=" in text:
        answer = text.split("id=")[-1].strip()[:1]
        answer = f"option {answer}" if answer in numbers else ""
        answer = f"option {letters.index(answer)}" if answer in letters else ""

        if answer in numbers_options:
            return [answer]
        elif answer in letters_options:
            return [numbers_options[letters_options.index(answer)]]
        else:
            answer = ""

    logger.debug(f"Sixth version of answer parsing: {answer}")

    if answer == "":
        if "zero" in text:
            answer = "option 0"
        elif "one" in text:
            answer = "option 1"
        elif "two" in text:
            answer = "option 2"
        elif "three" in text:
            answer = "option 3"
        elif "four" in text:
            answer = "option 4"
        else:
            answer = ""

        if answer in numbers_options:
            return [answer]
        else:
            answer = ""

    logger.debug(f"Seventh version of answer parsing: {answer}")

    if answer == "":
        if "0" in text:
            answer = "option 0"
        elif "1" in text:
            answer = "option 1"
        elif "2" in text:
            answer = "option 2"
        elif "3" in text:
            answer = "option 3"
        elif "4" in text:
            answer = "option 4"
        else:
            answer = ""

        logger.debug(f"Final version of answer parsing: {answer}")

        if answer in numbers_options:
            return [answer]
        else:
            answer = ""

    logger.debug(f"Eighth version of answer parsing: {answer}")

    if answer == "":
        if "a" in text:
            answer = "option 0"
        elif "b" in text:
            answer = "option 1"
        elif "c" in text:
            answer = "option 2"
        elif "d" in text:
            answer = "option 3"
        elif "e" in text:
            answer = "option 4"
        else:
            answer = ""

        logger.debug(f"Final version of answer parsing: {answer}")

        if answer in numbers_options:
            return [answer]
        else:
            answer = ""

    logger.debug(f"Ninth version of answer parsing: {answer}")

    answer = answer.replace("_", " ")
    for valid_answer in numbers_options:
        if valid_answer in answer:
            answer = valid_answer
            break

    logger.debug(f"Final version of answer parsing: {answer}")

    if answer not in numbers_options:
        return []
    else:
        return [answer]


def get_clip_data_from_video_data(video_data: list[str], sampled_indices: list[int], fps: float) -> dict:
    if len(video_data) == 0:
        raise ValueError("The video data must contain at least one sampled frame.")
    else:
        # create a dictionary with the tuples of start and end frame as keys
        # (the data must represent intervals of 1 second for each second of the video)
        # note that the keys can be anything since they are not used (i.e. dummy keys)
        # TODO decide whether to use the keys or not and be consistent in our framework
        keys = []
        for i in range(1, len(sampled_indices) + 1):
            start_key = int(round(sampled_indices[i - 1] / fps))
            end_key = int(round(sampled_indices[i] / fps)) if i < len(sampled_indices) else start_key + 1
            keys.append((start_key, end_key))
        logger.debug(f"Keys: {keys}")
        logger.debug(f"Len Keys: {len(keys)}")
        logger.debug(f"Len video_data: {len(video_data)}")
        logger.debug(f"Type video_data: {type(video_data)}")
        values = [video_data[max(min(int(round(i / fps)), len(video_data) - 1), 0)] for i in sampled_indices]
        logger.debug(f"Values: {values}")
        clip_data = dict(zip(keys, values))
        logger.debug(f"Clip data: {clip_data}")
    return clip_data


def parse_answer_json(text, keywords_in_priority_order, candidate_fun):
    # this function tries to parse the answer from the text by looking for JSON objects using the following priority:
    # 1. try to find exact JSON format matches
    # 2. try to find the best answer in the text after an opening curly bracket
    # 3. try to find the best answer in the text
    # 4. try to find the first single letter in the text
    # 5. use the first letter of the text

    # preprocess text
    text = text.strip().lower()
    text = text.replace("\n", "")

    # if text is empty, return None
    if len(text) == 0:
        return None

    # 1. try to find the best answer in the text within curly brackets first
    pattern = r'\{(.*?)\}'

    # find all JSON objects in the text
    matches = re.findall(pattern, text)
    logger.debug(f"Matches: {matches}")

    # get all potential json object matches
    json_objects = {}
    for match in matches:
        # prepare the match for json loading
        match = '{' + match + '}'

        # get all indices of double quotes or single quotes
        quote_indices = [i for i, c in enumerate(match) if c == '"' or c == "'"]
        quote_characters = [f"\\{c}" if c == '"' else c for i, c in enumerate(match) if c == '"' or c == "'"]

        # replace single quotes with double quotes for JSON loading
        match = match.replace("'", '"')

        # there should be exactly 4 double quotes
        # if there are more, then escape all the ones after the third one except the last one
        if len(quote_indices) > 4:
            # replace the characters at the indexes_to_escape in the match with the characters_to_escape
            indexes_to_escape = quote_indices[3:-1]
            characters_to_escape = quote_characters[3:-1]
            match = "".join([c if i not in indexes_to_escape else characters_to_escape.pop(0) for i, c in enumerate(match)])

        # try to load json, continue if fails
        try:
            data = json.loads(match)
            logger.debug(f"Loaded json: {data}")
        except json.JSONDecodeError:
            logger.debug(f"Failed to load json: {match}")
            continue

        # add the json object to the dict
        json_objects = json_objects | data
    logger.debug(f"JSON objects: {json_objects}")

    # if there is just one entry, return the value
    if len(json_objects) == 1:
        candidates = candidate_fun(list(json_objects.values())[0])
        if candidates:
            return candidates[0]

    # if there are multiple entries, try to find the best answer
    if len(json_objects) > 1:
        for keyword in keywords_in_priority_order:
            if keyword in json_objects.keys():
                candidates = candidate_fun(json_objects[keyword])
                if candidates:
                    return candidates[0]

    # 2. in case no match found, search for opening curly bracket and return the text after that
    splits = text.split('{')
    if len(splits) > 1:
        candidates = candidate_fun(splits[-1])
        if candidates:
            return candidates[0]

    # 3. in case no match found, search for the best answer in the text
    for keyword in keywords_in_priority_order:
        if keyword in text:
            candidates = candidate_fun(text.split(keyword)[-1])
            if candidates:
                return candidates[0]

    # 4. in case no match found, search for the first single letter in the text
    candidates = candidate_fun(text)
    if candidates:
        return candidates[0]

    # 5. in case no match found, return the first letter of the text (this is naive)
    candidates = candidate_fun(text[0])
    if candidates:
        return candidates[0]

    # no answer was found
    return None


def get_single_letter_candidates_from_text(text: str):
    if text is None or not isinstance(text, str) or len(text) == 0:
        return []
    pattern = r'\b[ABCDE]\b'
    single_letter_candidates = re.findall(pattern=pattern, string=text, flags=re.IGNORECASE)
    logger.debug(f"Found single letter candidates in text: {single_letter_candidates}")
    return single_letter_candidates


def get_single_number_candidates_from_text(text):
    if text is None:
        return []
    # if the json parsing already parsed integer successfully
    if isinstance(text, int):
        return [text]
    # if the json parsing already parsed a float, round to integer
    if isinstance(text, float):
        return [int(round(text))]
    # if the text is not a string, return empty list
    if not isinstance(text, str):
        return []
    pattern = r'\b\d\b'
    single_number_candidates = re.findall(pattern=pattern, string=text)
    single_number_candidates = [int(number) for number in single_number_candidates]
    logger.debug(f"Found single number candidates in text: {single_number_candidates}")
    return single_number_candidates


def letter_to_option_id(letter: str):
    letter_to_option = {
        "a": "option 0",
        "b": "option 1",
        "c": "option 2",
        "d": "option 3",
        "e": "option 4"
    }

    if letter.lower() not in letter_to_option:
        logger.warning(f"Letter {letter.lower()} not in letter_to_option mapping.")
        return []

    return [letter_to_option[letter.lower()]]


def parse_option_answer_naive(text: str):
    # naive selection of first character
    return letter_to_option_id(text.strip()[0])


def parse_free_form_answer_naive(text: str):
    # naive selection of the first sentence
    return [text.split(".")[0].strip()] if "." in text else [text.strip()]


def get_free_form_candidate_from_text(text: str):
    if text is None or not isinstance(text, str) or len(text) == 0:
        return []
    free_form_candidates = [text]
    logger.debug(f"Found free-form candidate in text: {free_form_candidates}")
    return free_form_candidates


def replace_c_with_camera_wearer(text):
    # replace "option C" with a placeholder since we do not want to replace the "C" in there
    option_c = "option C"
    placeholder = "whuut, is penny a freeloader!?? :O"
    text = text.replace(option_c, placeholder)

    # replace "C" with "the camera wearer"
    # like https://arxiv.org/pdf/2404.04346
    pattern = r'\bC\b'
    replaced_text = re.sub(pattern, 'the camera wearer', text, flags=re.IGNORECASE)

    # additional replacements from 24.05.2024
    # remove "the character" or "character" prefixes
    replaced_text = replaced_text.replace("the character the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("The character the camera wearer", "The camera wearer")
    replaced_text = replaced_text.replace("character the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("Character the camera wearer", "The camera wearer")

    # remove "the main character" or "main character" prefixes
    replaced_text = replaced_text.replace("the main character the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("The main character the camera wearer", "The camera wearer")
    replaced_text = replaced_text.replace("main character the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("Main character the camera wearer", "The camera wearer")

    # remove "the person" or "person" prefixes
    replaced_text = replaced_text.replace("the person the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("The person the camera wearer", "The camera wearer")
    replaced_text = replaced_text.replace("person the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("Person the camera wearer", "The camera wearer")

    # remove "the individual" or "individual" prefixes
    replaced_text = replaced_text.replace("the individual the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("The individual the camera wearer", "The camera wearer")
    replaced_text = replaced_text.replace("individual the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("Individual the camera wearer", "The camera wearer")

    # remove "the student" or "student" prefixes
    replaced_text = replaced_text.replace("the student the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("The student the camera wearer", "The camera wearer")
    replaced_text = replaced_text.replace("student the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("Student the camera wearer", "The camera wearer")

    # remove "the woman" or "woman" prefixes
    replaced_text = replaced_text.replace("the woman the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("The woman the camera wearer", "The camera wearer")
    replaced_text = replaced_text.replace("woman the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("Woman the camera wearer", "The camera wearer")

    # remove "the man" or "man" prefixes
    replaced_text = replaced_text.replace("the man the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("The man the camera wearer", "The camera wearer")
    replaced_text = replaced_text.replace("man the camera wearer", "the camera wearer")
    replaced_text = replaced_text.replace("Man the camera wearer", "The camera wearer")

    # replace the placeholder back with "option C"
    replaced_text = replaced_text.replace(placeholder, option_c)

    # make sure the first character is uppercase
    if len(replaced_text) > 0:
        replaced_text = replaced_text[0].upper() + replaced_text[1:]

    return replaced_text
