topiccards="""
Extract potential blog topics from the provided transcript by understanding the underlying motivations for why someone would want to write or publish blogs. Allow the degree of granularity for blog topics to be adjusted by the user, ranging from 1-10 specific topics.

Consider a "first principles" approach to uncover the main purposes and effects mentioned in the transcript. Analyze the sentiments, themes, and contexts expressed, particularly looking for deeper, implicit reasons behind the points discussed. From there, generate potential blog topics that capture both the surface information and the deeper motivations identified.

# Steps

1. **Analyze the Transcript**: Examine the transcript to identify key subjects, themes, or ideas mentioned.
2. **Understand Purposes Behind Writing**:
   - Isolate sections that indicate why a person would like to write or publish—e.g., sharing knowledge, storytelling, generating engagement, achieving expertise, or other motivations.
   - Account for "4th layer effects" or indirect impacts, such as those influencing career growth, brand presence, establishing authority, or catalysis of thought leadership.
3. **Extract and Refine Possible Topics**:
   - Use key topics and motivations from the transcript to generate potential blog post topics.
   - Ensure the topics align with both the explicit content and the "first principles" motivations behind why someone might share the information publicly.
   - Adjust the number of topics generated based on the user-specified granularity (ranging from 1 to 10 topics).

# Output Format

- Each card must start with "Topic N: " where N is the topic number
- Each card must be separated by TWO newlines
- Each topic must have a clear title and detailed content
        
        Example format:
        Topic 1: [Title]
        [Content for topic 1]

        Topic 2: [Title]
        [Content for topic 2]


Provide a list of suggested blog topics according to the user's preferred granularity (e.g., 1-10 topics). Each topic should:
- Be a concise, engaging title (3-10 words).
- Include a short sentence explaining why this topic would be valuable for publication based on its underlying motivation or purpose.

# Examples

**Transcript Excerpt**:
- "...and then I thought, sharing my journey from struggling beginner to understanding this topic deeply might really help others facing the same issues. It was a real struggle finding good meta-level strategies, so I think documenting some of these tips would save others a lot of time."

**User-Specified Granularity: 3 Topics**

**Generated Blog Topics**:
1. **"From Beginner to Expert: My Personal Learning Journey"**  
   *Sharing a relatable journey helps newcomers feel less isolated and offers hope based on lived experience.*
2. **"Meta Strategies for Overcoming Learning Roadblocks"**  
   *Focusing on high-level approaches provides valuable guidance for those struggling with common issues and challenges.*
3. **"Learning Challenges: How Documenting Mistakes Can Guide Others"**  
   *Documenting struggles and solutions is a way to share real tips that save time for others.*

(Note: Adjust the number of blog topics according to the granularity specified by the user.)

ONLY OUTPUT THE TOPICS, BEGIN:
"""