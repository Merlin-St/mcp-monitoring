# Gorilla Code Editor - Complete Documentation

This document compiles documentation from Gorilla's official support pages for the Code Editor, including API references, samples, and tutorials.

---

## Table of Contents

1. [Code Editor Overview](#code-editor-overview)
2. [Core API Functions](#core-api-functions)
3. [Code Editor Samples](#code-editor-samples)
4. [Best Practices](#best-practices)

---

## Code Editor Overview

The Gorilla Code Editor provides a "blank canvas" for programming custom experimental tasks with full JavaScript programming flexibility.

### Key Features

- **Minimal Interface**: Four core functions provide the foundation for any task
- **JavaScript Flexibility**: Full programming control for custom implementations
- **Third-party Integration**: Support for external libraries (jsPsych, etc.)
- **Template System**: Handlebars templates for flexible layout rendering
- **Layout System**: Screen resolution independence
- **State Machine**: Organized task logic flow

### Core Functions

1. `gorilla.ready()` - Initialize Gorilla subsystems
2. `gorilla.manipulation()` - Retrieve task configuration parameters
3. `gorilla.metric()` - Record experimental data
4. `gorilla.finish()` - Complete and end the task

### Tutorial Progression

1. **Hello World**: Basic task structure and commands
2. **Hello World Templates**: Page layout control
3. **Display Stimuli**: Stimulus presentation techniques
4. **Manipulations & Response Recording**: Task configuration and data collection
5. **State Machine**: Advanced task logic organization

### Advanced Capabilities

- jsPsych integration for standardized paradigms
- Custom game development
- Flexible event handling
- Comprehensive data tracking
- Timer sequences for precise timing control

---

## Core API Functions

### ready()

**Signature:**
```javascript
gorilla.ready(cb: () => any)
```

**Purpose:**
Main entry point for a task. Ensures Gorilla's subsystems are initialized and authentication is complete before executing the callback.

**Key Characteristics:**
- Guarantees that Gorilla is fully prepared before running subsequent code
- Prevents potential errors from running logic before system readiness

**Usage Warning:**
Do not run Gorilla functions before `ready()` completes, as they may fail.

**Example:**
```javascript
// BAD - Gorilla isn't ready yet
var mySetting = gorilla.retrieve('mySetting', 'blah');

// GOOD - Gorilla is ready
gorilla.ready(() => {
    var mySetting = gorilla.retrieve('mySetting', 'blah');
})
```

**Import Requirement (for Code Editor):**
```javascript
import gorilla = require("gorilla/gorilla");
```

---

### manipulation()

**Signature:**
```javascript
gorilla.manipulation(name: string, def?: any)
```

**Description:**
Retrieves the value of a manipulation. If the manipulation is not set, it returns the default value.

**Arguments:**
1. `name` (string, mandatory): The name of the manipulation
2. `def` (optional): Default value to return if manipulation is not set

**Usage Example:**
```javascript
var numTrials = gorilla.manipulation('numberOfTrials', 20);
```

**Key Points:**
- Useful for setting configurable parameters in an experiment
- Allows dynamic adjustment of task parameters
- Provides a fallback mechanism with default values
- Can be configured in the Task Builder or Code Editor's manipulations tab

**Example Use Cases:**
- Setting number of trials
- Configuring stimulus duration
- Adjusting difficulty levels
- Group assignment parameters

---

### metric()

**Signature:**
```javascript
gorilla.metric(results: any, key: string = ''): any
```

**Purpose:**
Upload a metric to the main data server

**Arguments:**
1. **results** (Mandatory): A dictionary of metrics with keys known to Gorilla
2. **key** (Optional): A string identifier for the specific row of metrics

**Available Metric Keys:**
- `trial_number` - Current trial number
- `spreadsheet_name` - Name of spreadsheet used
- `spreadsheet_row` - Row from spreadsheet
- `spreadsheet` - Spreadsheet data
- `screen_number` - Screen identifier
- `screen_name` - Screen name
- `zone_name` - UI zone name
- `zone_type` - UI zone type
- `response` - Participant's response
- `response_type` - Type of response
- `correct` - Correctness of response (boolean)
- `reaction_time` - Reaction time in milliseconds
- `reaction_onset` - When reaction timing started
- `timed_out` - Whether trial timed out (boolean)
- `attempt` - Attempt number
- `dishonest` - Flag for dishonest responses
- `x_coord` - X coordinate (for mouse/touch)
- `y_coord` - Y coordinate (for mouse/touch)

**Usage Example:**
```javascript
gorilla.metric({
    trial_number: 1,
    reaction_time: 450,
    response: 'yes',
    correct: true
});
```

**Key Characteristics:**
- No return value
- In Task Builder, keys must be from predefined list
- In Code Editor, custom keys can be created in Metrics tab
- Can include optional key for later retrieval

---

### store()

**Signature:**
```javascript
gorilla.store(key: string, value: any, global?: boolean)
```

**Purpose:**
Add information either to the task/questionnaires data storage or to the experiments data storage.

**Arguments:**
1. **key** (Mandatory): A string to uniquely identify the stored information
2. **value** (Mandatory): The value to be stored (can be any variable type: number, string, boolean, object)
3. **global** (Optional): A boolean to indicate the stored information's accessibility level (default: false)

**Key Characteristics:**
- Storage is unique to the current participant
- Persists across experiment logins
- Can be used to track participant progress or performance

**Storage Levels:**
- **Local** (default): Only available in the current task/questionnaire
- **Global** (when set to true): Accessible from anywhere in the experiment

**Usage Example:**
```javascript
gorilla.store('correctAnswers', 15, true);
```

**Pro Tip:**
Useful for longitudinal studies where participants log in multiple times across days/weeks/months.

---

### storeMany()

**Signature:**
```javascript
gorilla.storeMany(values: { [name: string]: any}, global?: boolean)
```

**Purpose:**
Stores multiple key-value pairs simultaneously.

**Usage Example:**
```javascript
gorilla.storeMany({
    'correctAnswers': 15,
    'incorrectAnswers': 5,
    'totalTime': 120
}, true);
```

---

### retrieve()

**Signature:**
```javascript
gorilla.retrieve(key: string, default?: any, global?: boolean)
```

**Purpose:**
Returns embedded data values unique to the current participant, which persist across experiment logins.

**Arguments:**
1. `key` (Mandatory): A string uniquely identifying the stored information
2. `default` (Optional): Value returned if no value is found for the key
3. `global` (Optional): Boolean indicating data accessibility level

**Key Characteristics:**
- Data is participant-specific
- Can be stored locally (task-level) or globally (experiment-level)
- Persists across multiple experiment sessions

**Usage Example:**
```javascript
var correct_answers = gorilla.retrieve('correctAnswers', 15, true);
```

---

### populate()

**Signature:**
```javascript
gorilla.populate(element: any, template: string, content?: any): any
```

**Arguments:**
1. **element** (Mandatory): A jQuery selector for the page element where the template will be loaded
   - Example: '#gorilla' (default Gorilla div)
   - Must be a valid jQuery selector

2. **template** (Mandatory): Name of the template to be loaded
   - Must match a template in the Templates tab of the Code Editor
   - Saved as a Handlebars file
   - Can be raw HTML or use Handlebars templating

3. **content** (Optional): An object of data to be inserted into the template
   - Used for dynamic content with Handlebars templating
   - Allows procedural HTML generation

**Basic Usage Example:**
```javascript
gorilla.populate('#gorilla', 'instructions');
```

**Advanced Usage Example (with dynamic content):**
```javascript
gorilla.populate('#gorilla', 'trial', {
    stimulus: 'cat.jpg',
    trialNumber: 5
});
```

**Handlebars Template Example:**
```html
<h1>Trial {{trialNumber}}</h1>
<img src="{{stimulus}}" />
```

---

### stimuliURL()

**Signature:**
```javascript
gorilla.stimuliURL(name: string)
```

**Purpose:**
Returns the URL to a named stimuli uploaded in the task's stimuli library.

**Arguments:**
1. **name** (mandatory): A string matching the full filename of an uploaded stimuli
   - Must match the exact filename, including file extension
   - If no matching stimuli is found, returns null

**Usage Example:**
```javascript
var fileURL = gorilla.stimuliURL('instructions.txt');
```

**Practical Application:**
```javascript
var imageURL = gorilla.stimuliURL('cat.jpg');
$('#stimulus').html('<img src="' + imageURL + '" />');
```

---

### shuffle()

**Signature:**
```javascript
gorilla.shuffle(array: [], seed?: number)
```

**Purpose:**
Randomly shuffles array elements.

**Arguments:**
1. `array` (Mandatory): Array to be shuffled
2. `seed` (Optional): Seed for reproducible shuffling

**Usage Example:**
```javascript
var trials = [1, 2, 3, 4, 5];
var shuffledTrials = gorilla.shuffle(trials);
```

---

### finish()

**Signature:**
```javascript
gorilla.finish(overrideURL?: string)
```

**Purpose:**
Immediately ends the current task, marking it as complete.

**Arguments:**
- Optional `overrideURL`: A string URL to redirect the participant after task completion

**Behavior:**
- Marks the current task as complete
- Advances to the next node in the experiment tree (by default)
- Optional URL parameter can redirect participant away from experiment

**Usage Examples:**

1. Basic task completion:
```javascript
gorilla.finish();
```

2. With redirect URL:
```javascript
gorilla.finish('https://example.com');
```

**Warning Notes:**
- Use with care to avoid disrupting participant experience
- Redirecting may prevent proper experiment completion tracking
- Consider using experiment's built-in onward URL functionality when possible

---

## Code Editor Samples

The following samples demonstrate various experimental paradigms implemented in Gorilla. Note that some samples were created using older versions of the Code Editor.

### 1. Corsi Block Tapping

**Description:**
Tests visuo-spatial short-term working memory. Participants are presented with blocks that flash in a sequence which needs to be repeated by tapping the correct blocks.

**Gorilla Functions Used:**
- manipulation
- ready
- run
- populate
- refreshLayout
- responsiveFrame
- store
- metric
- finish

**Key Concepts:**
- Create a display
- Store participants' responses and save them to metrics

---

### 2. CyberBall

**Description:**
Implementation of the Cyberball social exclusion paradigm.

**Gorilla Functions Used:**
- manipulation
- populate
- resourceURL
- run
- metric

---

### 3. Digit Span

**Description:**
Tests working memory by presenting number sequences that participants need to recollect and repeat in the same order.

**Gorilla Functions Used:**
- manipulation
- retrieve
- ready
- responsiveFrame
- populate
- refreshLayout
- store
- run

**Key Concepts:**
- Use gorilla manipulations
- Generate a display
- Store participants' responses

---

### 4. Headphone Check Task

**Description:**
Checks headphone functionality.

**Gorilla Functions Used:**
- ready
- populate
- store
- finish

---

### 5. JsPsych GoNoGo Example

**Description:**
Demonstrates how to set up a classic inhibition GoNoGo task from jsPsych in Gorilla.

**Gorilla Functions Used:**
- ready
- populate
- stimuliURL
- metric
- finish

**Key Concepts:**
- Set up a jsPsych task in Gorilla
- Display stimuli
- Save responses to metrics

---

### 6. Questionnaire Examples

**Description:**
Multiple approaches to creating questionnaires:

a. **Using HTML Forms** - Simple form-based questionnaires
b. **Basic Constructor with TypeScript/Handlebars** - Programmatic questionnaire generation
c. **Advanced Constructor** - Modular approach with reusable components

---

### 7. Reveal Demo

**Description:**
Economic psychology decision-making task.

**Gorilla Functions Used:**
- ready
- populate
- metric
- manipulation

---

### 8. Shopping Comparison Task

**Description:**
Online product comparison interface.

**Gorilla Functions Used:**
- ready
- populate
- finish

---

### 9. Stroop Task

**Description:**
Classic neuropsychological test of color/word interference.

**Gorilla Functions Used:**
- ready
- shuffle
- responsiveFrame
- populate
- refreshLayout
- startStopwatch
- stopStopwatch
- store
- getStopwatch
- metric
- finish
- run

**Key Concepts:**
- Set up and apply trials manipulations
- Randomize trials using gorilla.shuffle
- Store participants' responses and save them to metrics

---

### 10. Towers of Hanoi

**Description:**
Problem-solving puzzle task.

**Gorilla Functions Used:**
- manipulation
- retrieve
- store
- run
- metric

---

### 11. Virtual Chinrest Task

**Description:**
Tests participant viewing distance.

**Gorilla Functions Used:**
- ready
- populate
- metric
- store

---

### 12. iFrame Example

**Description:**
Demonstrates embedding external webpages.

**Gorilla Functions Used:**
- ready
- metric
- finish

---

## Best Practices

### Task Structure

1. **Always use gorilla.ready()**
   - Never execute Gorilla functions before ready() callback
   - Ensures all systems are initialized

2. **Manipulations for Configuration**
   - Use manipulations for parameters that may change between experiments
   - Always provide default values
   - Configure in the Manipulations tab

3. **Metrics for Data Collection**
   - Use standard metric keys when possible
   - Record data at appropriate granularity (per trial, per response)
   - Include timestamps and trial numbers for analysis

4. **Store/Retrieve for Persistence**
   - Use local storage for task-specific data
   - Use global storage for experiment-wide data
   - Useful for multi-session experiments

### Template Design

1. **Use Handlebars Templates**
   - Separate HTML from JavaScript logic
   - Create reusable templates
   - Use dynamic content injection

2. **Responsive Design**
   - Use responsiveFrame() for resolution independence
   - Test on multiple screen sizes
   - Consider mobile participants

### Data Management

1. **Structured Metrics**
   - Use consistent key names
   - Include metadata (trial numbers, timestamps)
   - Plan data structure before implementation

2. **Error Handling**
   - Validate participant responses
   - Handle edge cases
   - Provide clear error messages

3. **Progress Tracking**
   - Use store() to save progress
   - Allow participants to resume if needed
   - Track completion status

### Performance

1. **Minimize Server Calls**
   - Batch metrics when possible
   - Use local storage for temporary data
   - Only call metric() for final data

2. **Optimize Media Loading**
   - Preload stimuli when possible
   - Use appropriate file formats
   - Consider file sizes for slower connections

### Code Organization

1. **State Machine Pattern**
   - Organize complex tasks into states
   - Clear transitions between states
   - Easier to debug and maintain

2. **Modular Functions**
   - Break complex tasks into functions
   - Reusable code components
   - Easier testing and debugging

3. **Clear Variable Names**
   - Descriptive variable names
   - Consistent naming conventions
   - Comment complex logic

---

## Example: Complete Simple Task

Here's a complete example of a simple reaction time task:

```javascript
import gorilla = require("gorilla/gorilla");

// Global state
let state = {
    trialNumber: 0,
    maxTrials: 10,
    startTime: 0
};

gorilla.ready(function() {
    // Get configuration from manipulations
    state.maxTrials = gorilla.manipulation('numTrials', 10);

    // Load initial template
    gorilla.populate('#gorilla', 'instructions');

    // Start button handler
    $('#start-btn').on('click', function() {
        startTrial();
    });
});

function startTrial() {
    state.trialNumber++;

    if (state.trialNumber > state.maxTrials) {
        endExperiment();
        return;
    }

    // Show fixation
    gorilla.populate('#gorilla', 'fixation');

    // Wait random interval (500-1500ms)
    setTimeout(function() {
        showStimulus();
    }, 500 + Math.random() * 1000);
}

function showStimulus() {
    state.startTime = Date.now();

    // Show stimulus
    gorilla.populate('#gorilla', 'stimulus');

    // Response handler
    $('#response-btn').on('click', function() {
        recordResponse();
    });
}

function recordResponse() {
    let reactionTime = Date.now() - state.startTime;

    // Record metric
    gorilla.metric({
        trial_number: state.trialNumber,
        reaction_time: reactionTime
    });

    // Next trial
    setTimeout(startTrial, 500);
}

function endExperiment() {
    gorilla.populate('#gorilla', 'thankyou');

    setTimeout(function() {
        gorilla.finish();
    }, 2000);
}
```

**Templates:**

`instructions.html`:
```html
<div class="instructions">
    <h1>Reaction Time Task</h1>
    <p>Click as quickly as possible when you see the target.</p>
    <button id="start-btn">Start</button>
</div>
```

`fixation.html`:
```html
<div class="fixation">
    <h1>+</h1>
</div>
```

`stimulus.html`:
```html
<div class="stimulus">
    <h1>TARGET</h1>
    <button id="response-btn">Click Here!</button>
</div>
```

`thankyou.html`:
```html
<div class="thankyou">
    <h1>Thank you!</h1>
    <p>The experiment is now complete.</p>
</div>
```

---

## Additional Resources

### Official Links
- Code Editor Documentation: https://support.gorilla.sc/support/tools/legacy-tools/code-editor/code-editor
- Code Editor Samples: https://support.gorilla.sc/support/tools/legacy-tools/code-editor/code-editor-samples
- Gorilla API Reference: https://support.gorilla.sc/support/tools/legacy-tools/task-builder-1/gorilla-api

### Important Notes
- The Code Editor is part of Gorilla's Legacy Tooling
- Consider transitioning to Questionnaire Builder 2 and Task Builder 2 for new projects
- Some samples may require adaptation for newer versions

---

## Troubleshooting

### Common Issues

1. **Functions not working**
   - Ensure gorilla.ready() is called before any Gorilla functions
   - Check that imports are correct
   - Verify manipulation names match configuration

2. **Metrics not saving**
   - Check metric key names
   - Ensure proper data types
   - Verify network connectivity

3. **Templates not loading**
   - Verify template names match exactly
   - Check Templates tab in Code Editor
   - Ensure proper jQuery selectors

4. **Stimuli not found**
   - Check exact filename including extension
   - Verify file is uploaded in Stimuli tab
   - Use stimuliURL() to get correct path

---

Example 1: 

file exp_code
// The following jsPsych task is based off of the sample jspsych psiturk experiment
// https://github.com/jspsych/sample-jspsych-psiturk-experiment
// Integration with psiturk not included

// NB jsPsych has changed ALOT since then (the above example is many many years old)
// Notably, a lot has moved to the Timeline functionality
// When there's time, we'll try and create a more modern example
// I think many of the same principles apply

// In short, the basic principles of setting up a jsPsych task into Gorilla are
// 1) load up gorilla (gorilla.ready)
// 2) load a template to the screen (creating under Templates, loaded using gorilla.populate)
// 3) add all your jsPsych custom code in the body of the gorilla.ready callback function
// 4) in jspsych.init
// 4.1) link jspsych to the template you loading using display_element
// 4.2) link jspsych to gorilla.finish to end the task, via on_finish
// 4.3) link jspsych to gorilla.metric to upload metrica, via on_data_update
// 5) Add any stimuli files to the Uploads/Stimuli section
// 6) link jspsych to any stimuli files using gorilla.stimuliURL('name of stimuli')
// 7) Add any jspsych core files to Uploads/Resources and add them to the Head
// 8) Define any metrics in Experiment/Metrics
// 9) ...
// 10) Science???

// The main gorilla functionality
// You'll need this for gorilla.ready, populate, metric etc.
import gorilla = require("gorilla/gorilla");

// All the other jsPsych scripts we need have been uploaded as Resources and are pulled in as scripts in the 'Head'
// In the head, we use a custom handlebars helper 'resource' to tell Gorilla that it should look for the script
// in the Resources section of the task

// You don't need the line below (it will just stop the compiler from complaining!)
var jsPsych;

// gorilla.ready is a wrapper around document.ready
// It makes sure that all gorilla scripts have loaded, as well as any additional scripts loaded into the head
gorilla.ready(function() {
    // Load our basic template on to the page.
    // This is all gorilla needs to do to display the task
    // JsPsych will handle the generation of all elements to be displayed to the user
    // In the JsPsych.init function we will set the display_element to the id of the only div in 'exp'
    // It is this div jspsych will populate with task content
    gorilla.populate($('#gorilla'), 'exp', {});

    // Pretty much everything below this line is the original jsPsych task code
    // There's a couple of bits we've change to link things into Gorilla
    // We'll add a comment where this occurs
    var welcome_block = {
        type: "text",
        text: "Welcome to the experiment. Press any key to begin."
    };
    var instructions_block = {
        type: "text",
        text: "<p>In this experiment, a circle will appear in the center " + 
            "of the screen.</p><p>If the circle is <strong>blue</strong>, " + 
            "press the letter F on the keyboard as fast as you can.</p>" + 
            "<p>If the circle is <strong>orange</strong>, do not press " + 
            "any key.</p>" + 
            "<div class='left center-content'><img src=" + gorilla.stimuliURL("blue.png") + "></img>" + 
            "<p class='small'><strong>Press the F key</strong></p></div>" + 
            "<div class='right center-content'><img src=" + gorilla.stimuliURL("orange.png") + "></img>" + 
            "<p class='small'><strong>Do not press a key</strong></p></div>" + 
            "<p>Press any key to begin.</p>",
        timing_post_trial: 2000
    };
    // In the code above, we pull in the URL for the blue and orange circles using gorilla.stimuliURL
    // The two images have been uploaded as stimuli (under the uploads section on the left)
    // Then we use the name of the stimuli to retrieve the right url
    
    /* stimulus block */
    var test_stimuli = [{
        image: gorilla.stimuliURL("blue.png"),
        data: {
            response: 'go'
        }
    }, {
        image: gorilla.stimuliURL("orange.png"),
        data: {
            response: 'no-go'
        }
    }];
    
    var all_trials = jsPsych.randomization.repeat(test_stimuli, 10, true);
    var post_trial_gap = function() {
        return Math.floor(Math.random() * 1500) + 750;
    };
    
    var test_block = {
        type: "single-stim",
        stimuli: all_trials.image,
        choices: ['F'],
        timing_stim: 1500,
        timing_response: 1500,
        timing_post_trial: post_trial_gap,
        data: all_trials.data
    };
    
    /* debrief block */
    var debrief_block = {
        type: "text",
        text: function() {
            return "<p>Your average response time was <strong>" + 
            getAverageResponseTime() + "ms</strong>. Press " + 
            "any key to complete the experiment. Thank you!</p>";
        }
    };
    
    function getAverageResponseTime() {
        var trials = jsPsych.dataAPI.getTrialsOfType('single-stim');
        var sum_rt = 0;
        var valid_trial_count = 0;
        for (var i = 0; i < trials.length; i++) {
            if (trials[i].response == 'go' && trials[i].rt > -1) {
                sum_rt += trials[i].rt;
                valid_trial_count++;
            }
        }
        return Math.floor(sum_rt / valid_trial_count);
    }
    
    /* define experiment structure */
    var experiment_blocks = [];
    experiment_blocks.push(welcome_block);
    experiment_blocks.push(instructions_block);
    experiment_blocks.push(test_block);
    experiment_blocks.push(debrief_block);
    /* start the experiment */
    
    // This is where the most changes have occured, to wire everything into Gorilla
    // First, we tell jsPsych where to display the contents of the task using the display_element parameter
    // This is set to the element with the id 'jspsych-target', which can be found in our HTML template 'exp'
    // Next, we tell jsPsych that, when the task finishes, it should run gorilla.finish
    // gorilla.finish ends the current task and tells Gorilla to move the participant to the next node in the tree
    // Then, we tell jsPsych that whenever we want to upload data, call gorilla.metric
    // gorilla.metric expects an object of key/value pairs, where the key's match up to keys listed in your Metrics
    // (found on the left hand side under 'Experiment')
    jsPsych.init({
        display_element: $('#jspsych-target'),
        experiment_structure: experiment_blocks,
        on_finish: function() {
           gorilla.finish();
        },
        on_data_update: function(data) {
            gorilla.metric(data);
        }    
    });
});
        
file exp
<div class="container">
    <div id='jspsych-target'>
        
    </div>
</div>

file style
@import "/style/style.less";

/* 
 * CSS for jsPsych experiments.
 *
 * This stylesheet provides minimal styling to make jsPsych 
 * experiments look polished without any additional styles.
 *
 */

/*
 *
 * fonts and type
 *
 */
 
@import url(//fonts.googleapis.com/css?family=Open+Sans:400italic,700italic,400,700);

html {
 font-family: 'Open Sans', 'Arial', sans-serif;
 font-size: 18px;
 line-height: 1.6em;
}

p {
    clear:both;
}

.very-small {
    font-size: 50%;
}

.small {
    font-size: 75%;
}

.large {
    font-size: 125%;
}

.very-large {
    font-size: 150%;
}

/*
 *
 * Classes for changing location of things
 *
 */
 
.left {
    float: left;
}

.right {
    float: right;
}

.center-content {
    text-align: center;
}

/*
 *
 * Form elements like input fields and buttons
 *
 */

input[type="text"] {
    font-family: 'Open Sans', 'Arial', sans-sefif;
    font-size: 14px;
}

button {
    padding: 0.5em;
    background-color: #eaeaea;
    border: 1px solid #eaeaea;
    color: #333;
    font-family: 'Open Sans', 'Arial', sans-serif;
    font-size: 14px;
    cursor: pointer;
}

button:hover {
    border:1px solid #ccc;
}

/*
 *
 * Container holding jsPsych content
 *
 */


.jspsych-display-element {
    width: 800px;
    margin: 50px auto 50px auto;
}

/*
 *
 * PLUGIN: jspsych-single-stim
 *
 */
 
#jspsych-single-stim-stimulus {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

/*
 *
 * PLUGIN: jspsych-survey-text
 *
 */
 
 .jspsych-survey-text {
     margin: 0.25em 0em;
 }
 
 .jspsych-survey-text-question {
     margin: 2em 0em;
 }

file head
<script src="{{resource 'jspsych.js' config=content.config}}" type="text/javascript"></script>
<script src="{{resource 'jspsych-call-function.js' config=content.config}}" type="text/javascript"></script>
<script src="{{resource 'jspsych-single-stim.js' config=content.config}}" type="text/javascript"></script>
<script src="{{resource 'jspsych-text.js' config=content.config}}" type="text/javascript">></script>

Example 2
main file
// This task demonstrates how to use HTML 5 Forms and Bootstrap forms to create a Questionnaire
// A submit button at the end of the questionnaire will collate the data together and upload it to Gorilla

// If you want to know more about the different HTML Form elements take a look at these online guides
// https://www.w3schools.com/html/html_forms.asp
// https://developer.mozilla.org/en-US/docs/Learn/HTML/Forms

// If you want to know more about how Bootstrap implements form elements, go here:
// https://getbootstrap.com/docs/3.3/css/#forms

// import the basic gorilla functionality
import gorilla = require("gorilla/gorilla");

// this is our wrapper around $(document).ready
// this function is called when the page is ready
gorilla.ready(()=>{
   
    // populate the div with id 'gorilla' with the contents of the template 'basicForm' or 'basicBootstrapForm', with no additional variables
    // a div with id 'gorilla' is the default that you recieve on every page
    // change the second argument to change which template you see
    gorilla.populate('#gorilla', 'basicForm', {});
    //gorilla.populate('#gorilla', 'basicBootstrapForm', {});
   
    // Now we need to bind our submit button 
    $('#myForm').submit(function(event){
        // First, we need to prevent the default action that usually occurs on a form submit button
        // we want to handle collation and submission of data ourselves
        event.preventDefault();
        // Collate all of our form elements
        var x = document.getElementsByTagName("input");
        for (var i = 0; i < x.length; i++) {
            
            // This is quite a simple implementation
            // It will upload the data from every single element
            // For instance, with radio buttons and checkboxes, you probably only want the checked elements
            // Using x[i].type to see if its a radio or checkbox, you can do some filtering to remove the unchecked elements
            var name = x[i].name;
            var value = x[i].value;
            var checked = (x[i].checked ? '1' : '');
            
            // upload the result to the metrics
            // gorilla.metric takes as an argument an object of key-value pairs
            // In the 'Metrics' tools, we've set three keys: name, value and checked
            // We've also given them titles for our spreadsheet
            gorilla.metric({
                name: name,
                value: value,
                checked: checked,
            });
                
        }
        
        // gorilla.finish() ends the current task and moves on to the next node in the experiment tree
        gorilla.finish();
    });
});
Readme file
/* README 
This file contents a simple overview of how coding tasks in Gorilla are constructed compared to the 
common structure of web applications which may be faimilar to you.

--------
**CODE**
--------

In these files, we build the functionality and interactivity of our web task.
Here you place code that you may normally wrap in <script> tags.

Note that you don't need to wrap any of this code in <script> tags - Gorilla does that bit for you.

Its in these files that most of the code unique to Gorilla will appear.
That said, that's likely to be only a couple of lines - most of your web task will standard web application code.

In these files you can use Typescript (a typed superset of Javascript - most of the time, if you type normal 
Javascript, it will work), and jQuery.  
In brief, typescript allows us to create the functionality of our web application and jQuery allows 
us to link that functionality to elements on our page (our 'templates')

To learn more about Typescript: https://www.typescriptlang.org/docs/home.html
To learn more about jQuery: http://learn.jquery.com/

*********
At the top of these code files, you'll begin by including any other files that you may need.
These are other code files that you have written in this project, or files provided by Gorilla, 
like 'gorilla' and 'stateMachine'.
These **aren't** any files you may have uploaded to Resources or links you've included in the Head -
Gorilla will automatically include these for you, though you may need to add some additional code for
the editor to recognise the typing.

You'll probably need to put most of your code inside of gorilla.ready

gorilla.ready(()=>{
    ** Your code goes here **
})

This is our wrapper around the standard $(document).ready() function, used to determine when the page 
is ready for scripts to run and function correctly.
If you have any code that doesn't interact with or need to be bound to any elements on your page - like
preparing arrays of variables or helper functions, these can go outside of gorilla.ready

You'll need to use gorilla.populate to add content to your screen

gorilla.populate(**jQuery selector for element to add content to**, **name of template**, **object of variables**);

Most of the time, you'll want to add content to the main div we give you on the page, which has the id gorilla
and you probably won't need to add any variables to the page

Finally, you'll need to use gorilla.metric to upload any data you want collected to the Gorilla database

gorilla.metric(
    **object of key-value pairs**
)

In the 'Metrics' tab under the header 'Experiment', you set up the keys that you want Gorilla to store - these 
keys will need to match the keys you use in gorilla.metric.
A number of basic metrics are given to you by default, such as created time, participant ID.
Look at your download to see!

-------------
**TEMPLATES**
-------------

In these files you'll structure the content of your web application.
The names of these templates will likely be used in your gorilla.populate call
These templates contain content that you would normally find inside the <body> tags of a HTML webpage.
Note that you don't need to place <body> tags anywhere, nor do you need to include the commonly seen <HTML> tag
that appears at the very top of a webpage.

These files generally include no content unique to Gorilla - any standard HTML5 code will work in these files.
You can add classes and id's to elements as normal and add styling to them directly.

These templates are actually .handlebars files, so as well as standard HTML5 you can also use all the richness 
afforded by Handlebars and it's dynamic web content capabilities.

To learn more about HTML5: https://developer.mozilla.org/en-US/docs/Web/Reference
To learn more about Handlebars: http://handlebarsjs.com/

---------
**STYLE**
---------
USER WARNING - Style files (css) can take a long time to compile.  When you're adding new styling, make sure
the indicator at the bottom of the page has fully cycled through compiling and turned green.
Otherwise, your styling may not appear on your page.

In these files, you'll add any css styling that you want to attach to certain elements, classes or ids 
in your templates.
This file works exactly like a normal css style file would.
Don't forget to add the line

@import '/style/style.less';

To include the basic styling of Gorilla!

To learn more about CSS: https://developer.mozilla.org/en-US/docs/Learn/CSS/Introduction_to_CSS

*/

basicbootstrapfrom file
<div class="container">
    <div class="row">
        <div class="col-xs-6">
            <form id="myForm"> 
                <h1>Enter some information!</h1>
                </br>
                <p><strong>Enter your name</strong></p>
                <div class="form-group">
                    <label for="first-name">First name:</label>
                    <input type="text" class="form-control" id="first-name" name="firstname">
                </div>
                <div class="form-group">
                    <label for="last-name">Last name:</label>
                    <input type="text" class="form-control" id="last-name" name="lastname">
                </div>
                </br>
                <p><strong>Left handed, right handed or ambidextrous?</strong></p>
                <div class="form-group">
                    <div class="radio">
                        <label>
                            <input type="radio" id="handedness-left" name="handedness" value="left" >
                            Left
                        </label>
                    </div>
                    <div class="radio">
                        <label>
                            <input type="radio" id="handedness-right" name="handedness" value="right"> 
                            Right
                        </label>
                    </div>
                    <div class="radio">
                        <label>
                            <input type="radio" id="handedness-ambi" name="handedness" value="ambidextrous">
                            Ambidextrous
                        </label>
                    </div>
                </div>
                </br>
                <p><strong>Select some colours</strong></p>
                <div class="form-group">
                    <div class="checkbox">
                        <label>
                            <input type="checkbox" id="colour-red" name="color" value="red">
                            Red
                        </label>
                    </div>
                    <div class="checkbox">
                        <label>
                            <input type="checkbox" id="colour-blue" name="color" value="blue">
                            Blue
                        </label>
                    </div>
                    <div class="checkbox">
                        <label>
                            <input type="checkbox" id="colour-green" name="color" value="green">
                            Green
                        </label>
                    </div>
                    <div class="checkbox">
                        <label>
                            <input type="checkbox" id="colour-yellow" name="color" value="yellow">
                            Yellow
                        </label>
                    </div>
                </div>
                </br>
                <input type="submit" style="margin-top: 10px;">
            </form>
    </div>
    </div>
</div>

basicform file

<div class="container">
    <form id="myForm"> 
        <h1>Enter some information!</h1>
        </br>
        <p><strong>Enter your name</strong></p>
        <div>
            <label for="first-name">First name:</label>
            <input type="text" id="first-name" name="firstname">
        </div>
        <div>
            <label for="last-name">Last name:</label>
            <input type="text" id="last-name" name="lastname">
        </div>
        </br>
        <p><strong>Left handed, right handed or ambidextrous?</strong></p>
        <div>
            <!--Tip: A label can be bound to an element either by using the "for" attribute, or by placing the element inside the <label> element.
            https://www.w3schools.com/tags/tag_label.asp -->
            <input type="radio" id="handedness-left" name="handedness" value="left" >
            <label for="handedness-left">Left</label>
            <input type="radio" id="handedness-right" name="handedness" value="right"> 
            <label for="handedness-right">Right</label>
            <input type="radio" id="handedness-ambi" name="handedness" value="ambidextrous">
            <label for="handedness-ambi">Ambidextrous</label>
        </div>
        </br>
        <p><strong>Select some colours</strong></p>
        <div>
            <input type="checkbox" id="colour-red" name="color" value="red">
            <label for="colour-red">Red</label>
            <input type="checkbox" id="colour-blue" name="color" value="blue">
            <label for="colour-blue">Blue</label>
            <input type="checkbox" id="colour-green" name="color" value="green">
            <label for="colour-green">Green</label>
            <input type="checkbox" id="colour-yellow" name="color" value="yellow">
            <label for="colour-yellow">Yellow</label>
        </div>
        </br>
        <input type="submit" style="margin-top: 10px;">
    </form>
</div>

style file

@import '/style/style.less';

metrics.md file
name
value
checked

Example 3

main file
// This task demonstrates how to use Typescript and Handlebars templates to custom build a Questionnaire using
// basic HTML 5 form elements
// It will include much the same functionality as the 'BasicQuestionnaireUsingForms' code task, except that,
// rather than using a premade questionnaire in a template, we'll dynamically build the questionnaire
// from smaller elements

// To do this, we'll demonstrate how to use Typescript interfaces, to make sure you're always including
// all the data necessary in an object
// We'll also demonstrate how to make use of Handlebars variables and Handlebar Helpers such as if and each

// For the sake of simplicity and brevity, we'll only demonstrate a couple of form elements (text and radio)

// If you want to know more about the different HTML Form elements take a look at these online guides
// https://www.w3schools.com/html/html_forms.asp
// https://developer.mozilla.org/en-US/docs/Learn/HTML/Forms

// To learn more about Typescript interfaces: https://www.typescriptlang.org/docs/handbook/interfaces.html
// To learn more about Handlebars: http://handlebarsjs.com/expressions.html
// ------------------------------------------------------------------------------

// import the basic gorilla functionality
import gorilla = require("gorilla/gorilla");

// First, we're going to create an object of key-value pairs which we'll use as identifiers for our
// Questionnaire elements
// Creating this will make assigning these identifies quick and less prone to error

var QuestionnaireKeys = {
    text: 'text',
    radioset: 'radioset',
}

// ********************
// Everything between the line of asterixes could be put into a seperate code file
// or several seperate code files: one for the interfaces and definitions
// another to actually build your questionnaire
// This would make the individual files much easier to handle and work with, making it
// easier to scale the task up to handle more diverse and complicated questionnaire
// or display multiple different questionnaires one after the other

// See the code task QuestionnaireConstructorAdvanced for an implementation of this

// Here we create some interfaces for our Questionnaire elements

// when we create a variable of the type QuestionnaireText, we now know that it must contain the properties
// type, label, id and name, all of which are strings
// If we don't give it one of these, the compiler will remind us that it needs to be there
// We also indicate (using a question mark) that it contains an optional variable title
// it won't complain if we don't give it this variable
interface QuestionnaireText {
    type: string;
    label: string;
    id: string;
    name: string;
    title?: string;
}

// Radio buttons on a questionnaire usually appear in a set
// So we're going to define an interface whose only property is an array of type QuestionnaireRadio
interface QuestionnaireRadioSet {
    type: string;
    elements: QuestionnaireRadio[];
    title?: string;
}

// In QuestionnaireRadio, we define the properties we'll need to create our radio buttons
interface QuestionnaireRadio {
    label: string;
    id: string;
    name: string;
    value: string;
}

// Now, we're going to create the array which indicates what we want our Questionnaire to be built out of

var Questionnaire: any = [];

var FirstName: QuestionnaireText = {
    type: QuestionnaireKeys.text,
    label: 'First Name:',
    id: 'first-name',
    name: 'firstname',
    title: 'Enter your name!'
};

Questionnaire.push(FirstName);

var LastName: QuestionnaireText = {
    type: QuestionnaireKeys.text,
    label: 'Last Name:',
    id: 'last-name',
    name: 'lastname'
};

Questionnaire.push(LastName);

// Create our radio button set
var Handedness: QuestionnaireRadioSet = {
    type: QuestionnaireKeys.radioset,
    elements: [],
    title: 'Left handed, right handed or ambidextrous?',
}

var HandednessLeft: QuestionnaireRadio = {
    label: 'Left',
    id: 'handedness-left',
    name: 'handedness',
    value: 'left',
}

Handedness.elements.push(HandednessLeft);

var HandednessRight: QuestionnaireRadio = {
    label: 'Right',
    id: 'handedness-right',
    name: 'handedness',
    value: 'right',
}

Handedness.elements.push(HandednessRight);

var HandednessAmbi: QuestionnaireRadio = {
    label: 'Ambidextrous',
    id: 'handedness-ambi',
    name: 'handedness',
    value: 'ambi',
}

Handedness.elements.push(HandednessAmbi);

Questionnaire.push(Handedness);
// ********************


// this is our wrapper around $(document).ready
// this function is called when the page is ready
gorilla.ready(()=>{
   
    // populate the div with id 'gorilla' with the contents of the template 'basicForm', with no additional variables
    // a div with id 'gorilla' is the default that you recieve on every page
    gorilla.populate('#gorilla', 'basicForm', {});
    
    // Now that we have our basic form in place, we're going to start adding our questionnaire elements to it
    for(var i = 0; i < Questionnaire.length; i++){
        // we begin by adding a new div inside our dynamic-form div
        var elementID = 'element-' + i;
        $('.dynamic-form').append('<div id="' + elementID + '"></div></br>');
        
        // now depending on what our Questionnaire element is, we need to populate this div with different content
        switch(Questionnaire[i].type){
            case QuestionnaireKeys.text:
                gorilla.populate('#' + elementID, 'questionnaireText', Questionnaire[i]);
                break;
                
            case QuestionnaireKeys.radioset:
                gorilla.populate('#' + elementID, 'questionnaireRadioset', Questionnaire[i]);
                break;
        }
    }
   
    // Now we need to bind our submit button 
    $('#myForm').submit(function(event){
        // First, we need to prevent the default action that usually occurs on a form submit button
        // we want to handle collation and submission of data ourselves
        event.preventDefault();
        // Collate all of our form elements
        var x = document.getElementsByTagName("input");
        for (var i = 0; i < x.length; i++) {
            
            // This is quite a simple implementation
            // It will upload the data from every single element
            // For instance, with radio buttons and checkboxes, you probably only want the checked elements
            // Using x[i].type to see if its a radio or checkbox, you can do some filtering to remove the unchecked elements
            var name = x[i].name;
            var value = x[i].value;
            var checked = (x[i].checked ? '1' : '');
            
            // upload the result to the metrics
            // gorilla.metric takes as an argument an object of key-value pairs
            // In the 'Metrics' tools, we've set three keys: name, value and checked
            // We've also given them titles for our spreadsheet
            gorilla.metric({
                name: name,
                value: value,
                checked: checked,
            });
                
        }
        
        // gorilla.finish() ends the current task and moves on to the next node in the experiment tree
        gorilla.finish();
    });
});

readme structure file 
/* README STRUCTURE
This file contents a simple overview of how coding tasks in Gorilla are constructed compared to the 
common structure of web applications which may be faimilar to you.
If you want to know more about the specfics of this tasks implementation, look at the file 'README'

--------
**CODE**
--------

In these files, we build the functionality and interactivity of our web task.
Here you place code that you may normally wrap in <script> tags.

Note that you don't need to wrap any of this code in <script> tags - Gorilla does that bit for you.

Its in these files that most of the code unique to Gorilla will appear.
That said, that's likely to be only a couple of lines - most of your web task will standard web application code.

In these files you can use Typescript (a typed superset of Javascript - most of the time, if you type normal 
Javascript, it will work), and jQuery.  
In brief, typescript allows us to create the functionality of our web application and jQuery allows 
us to link that functionality to elements on our page (our 'templates')

To learn more about Typescript: https://www.typescriptlang.org/docs/home.html
To learn more about jQuery: http://learn.jquery.com/

*********
At the top of these code files, you'll begin by including any other files that you may need.
These are other code files that you have written in this project, or files provided by Gorilla, 
like 'gorilla' and 'stateMachine'.
These **aren't** any files you may have uploaded to Resources or links you've included in the Head -
Gorilla will automatically include these for you, though you may need to add some additional code for
the editor to recognise the typing.

You'll probably need to put most of your code inside of gorilla.ready

gorilla.ready(()=>{
    ** Your code goes here **
})

This is our wrapper around the standard $(document).ready() function, used to determine when the page 
is ready for scripts to run and function correctly.
If you have any code that doesn't interact with or need to be bound to any elements on your page - like
preparing arrays of variables or helper functions, these can go outside of gorilla.ready

You'll need to use gorilla.populate to add content to your screen

gorilla.populate(**jQuery selector for element to add content to**, **name of template**, **object of variables**);

Most of the time, you'll want to add content to the main div we give you on the page, which has the id gorilla
and you probably won't need to add any variables to the page

Finally, you'll need to use gorilla.metric to upload any data you want collected to the Gorilla database

gorilla.metric(
    **object of key-value pairs**
)

In the 'Metrics' tab under the header 'Experiment', you set up the keys that you want Gorilla to store - these 
keys will need to match the keys you use in gorilla.metric.
A number of basic metrics are given to you by default, such as created time, participant ID.
Look at your download to see!

-------------
**TEMPLATES**
-------------

In these files you'll structure the content of your web application.
The names of these templates will likely be used in your gorilla.populate call
These templates contain content that you would normally find inside the <body> tags of a HTML webpage.
Note that you don't need to place <body> tags anywhere, nor do you need to include the commonly seen <HTML> tag
that appears at the very top of a webpage.

These files generally include no content unique to Gorilla - any standard HTML5 code will work in these files.
You can add classes and id's to elements as normal and add styling to them directly.

These templates are actually .handlebars files, so as well as standard HTML5 you can also use all the richness 
afforded by Handlebars and it's dynamic web content capabilities.

To learn more about HTML5: https://developer.mozilla.org/en-US/docs/Web/Reference
To learn more about Handlebars: http://handlebarsjs.com/

---------
**STYLE**
---------
USER WARNING - Style files (css) can take a long time to compile.  When you're adding new styling, make sure
the indicator at the bottom of the page has fully cycled through compiling and turned green.
Otherwise, your styling may not appear on your page.

In these files, you'll add any css styling that you want to attach to certain elements, classes or ids 
in your templates.
This file works exactly like a normal css style file would.
Don't forget to add the line

@import '/style/style.less';

To include the basic styling of Gorilla!

To learn more about CSS: https://developer.mozilla.org/en-US/docs/Learn/CSS/Introduction_to_CSS

*/

readme file
/* README
This file contains an overview of some of the content specific to this task
For a more general overview of how code tasks are implemented in Gorilla checkout 'README-STRUCTURE'

--------
**CODE**
--------
In this code, we make use of Typescript interfaces
"One of TypeScript’s core principles is that type-checking focuses on the shape that values have. 
This is sometimes called “duck typing” or “structural subtyping”. 
In TypeScript, interfaces fill the role of naming these types, and are a powerful way of defining 
contracts within your code as well as contracts with code outside of your project."
From https://www.typescriptlang.org/docs/handbook/interfaces.html

In brief, normal javascript is very *very* flexible when it comes to defining variables and their types
Typescript allows us to add more rigor and structure to that system so that we can define complicated
variables and structures that must satisfy certain conditions.
This is very powerful when you are likely to be creating many instances of the same type of variable
(a process which is easily prone to error) where errors can cause the code to faile (and be time consuming
to debug)

For example, consider the first interface we define QuestionnaireText and the first instance of its use

----
interface QuestionnaireText {
    type: string;
    label: string;
    id: string;
    name: string;
    title?: string;
}

var FirstName: QuestionnaireText = {
    type: QuestionnaireKeys.text,
    label: 'First Name:',
    id: 'first-name',
    name: 'firstname',
    title: 'Enter your name!'
};
----

Here, we are saying that any variable of type QuestionnaireText must have the following properties
    type
    label
    id
    name
Additionally it has the optional property
    title

Whenever we create a variable which we give the type QuestionnaireText, if we don't include some of the
mandatory properties or give them the wrong value type i.e. try to give label a boolean value or integer value
the compiler will flag it as an error
Introducing this kind of strict typing prevents us from making mistakes that would cause our web task
to function incorrectly

This is a very powerful system, giving us a lot of flexibility and capabilities.
For example, consider the QuestionnaireRadioset

----
interface QuestionnaireRadioSet {
    type: string;
    elements: QuestionnaireRadio[];
    title?: string;
}
----

Here we say that any variable of type QuestionnaireRadioSet must contain a property elements which is itself
an array of type QuestionnaireRadio i.e. an array of objects of type QuestionnaireRadio.

By creating these kind of interfaces, we can quickly make sure we are always adding the right properties to
objects when we create them, preventing some of the more common coding mistakes from occuring.

-------------
**TEMPLATES**
-------------

In the templates, we introduce handlebars expressions and handlebars helpers.

Going back to the code, consider our first call to gorilla.populate

gorilla.populate('#' + elementID, 'questionnaireText', Questionnaire[i]);

This is the same form as we've seen previously, except now we are using that third arguement to pass in an object
to gorilla.populate.

In the case of our first questionnaire element, this object will be 
{
    type: QuestionnaireKeys.text,
    label: 'First Name:',
    id: 'first-name',
    name: 'firstname',
    title: 'Enter your name!'
}

Compare this to our template "questionnaireText"

{{#if title}}<p><strong>{{title}}</strong></p>{{/if}}
<label for="{{id}}">{{label}}</label>
<input type="text" id="{{id}}" name="{{name}}">

The sections in double curly braces "{{title}}" are handlebars expressions.
"This expression means "look up the title property in the current context""
From http://handlebarsjs.com/expressions.html

In this case the "context" is the object that we have just passed into gorilla.populate.
In the place of {{id}} we will put the value of id from the object.
In the place of {{name}} we will put the value of name from the object.

Using this, we've created a very simple, barebones template that we can reuse again and again to create
a variety of different content!

This template also demonstrates a handlebars helper if.
These helpers are actually block helpers which have some special behaviour when it comes to context
Learn more about block helpers: http://handlebarsjs.com/builtin_helpers.html

In this case we are saying if the current context has a value for the key title, we're going to do something
This ties in with title being an optional property in our interface.
If we don't include this property, then this section of the template will not display.

Another handlebars helper is each
{{#each elements}}
    **create some content**
{{/each}}

Here we're saying that for each object in elements create some content.
There's another important concept that comes in here, which is context.
Within the each statement, the context is the contents of the current elements object.
We know from our interface that elements is an array of QuestionnaireRadio objects.
So, the context for this section of the template will match the properties we know are found
in a QuestionnaireRadio object, which are 
{
    label: string;
    id: string;
    name: string;
    value: string;
}

Context can be quite a tricky subject when you first get started.
If you find that content isn't displaying as you'd expect when using #each or other block helpers 
check that you've understood the context these helpers are working with.
It may be they're trying to access properties which aren't available to them.


*/

questionniarradio file
{{#if title}}<p><strong>{{title}}</strong></p>{{/if}}
{{#each elements}}
    <input type="radio" id="{{id}}" name="{{name}}" value="{{value}}" >
    <label for="{{id}}">{{label}}</label>
{{/each}}

questionnair text file

<!-- Into this template, we'll be passing an object of type QuestionnaireText
This will contain an id, a label and a name (we don't need to use the type here
It may also contain a title element
We use the Handlebars Helper if to control whether this element displays
-->
{{#if title}}<p><strong>{{title}}</strong></p>{{/if}}
<label for="{{id}}">{{label}}</label>
<input type="text" id="{{id}}" name="{{name}}">

basic form file
<div class="container">
    <form id="myForm">
        <h1>Enter some information!</h1>
        <div class="dynamic-form">
            
        </div>
        <input type="submit" style="margin-top: 10px;">
    </form>
</div> 

style file
@import '/style/style.less';


Example 4
Readme file
/* README
This file contains an overview of some of the content specific to this task
For a more general overview of how code tasks are implemented in Gorilla checkout 'README-STRUCTURE'

-----------
**STRUCTURE
-----------

Compared to the basic version, you can see that the code has been divided up into three different files
main - this is where the main task code is implemented, with the contents being displayed to the screen
and metrics being recorded
questionnaire - this is essentially a definitions file, where we create all the interfaces for our 
questionnaire elements
basicQuestionnaire - this is where we create our first questionnaire and create a function to return it

To access the contents of these files we need to two things
1) In the file we want to access, any content we want accessible to other files needs to be prepended with
export
2) In the file where we want to access, we need to import the file to be accessed
3) To access the contents we need to use the dot notation

For example, we want to make the variable QuestionnaireKeys from questonnaire accessible in other files
We start by prepending its definition with export
export QuestionnaireKeys...

Next, in main we import the file questionnaire

import questionnaire = require('questionnaire')

Finally, we use the dot notation to access QuestionnaireKeys

questionnaire.QuestionnaireKeys

Structuring our contents in this way make the individual files much easier to work with.
They are smaller and its easy to see that each file has a clear purpose
It also makes the code task easier to scale.
You could create more complicated questionnaires in seperate files, while keeping your main file short
and handling only how these questionnaire display to the user and how they are interacted with
We then also know very easily where to go if we need to change something, like add a new questionnaire
element definition, rather than having to scroll through one massive file

NB. Below is a repeat of the descriptive contents found in QuestionnaireConstructorBasic
--------
**CODE**
--------
In this code, we make use of Typescript interfaces
"One of TypeScript’s core principles is that type-checking focuses on the shape that values have. 
This is sometimes called “duck typing” or “structural subtyping”. 
In TypeScript, interfaces fill the role of naming these types, and are a powerful way of defining 
contracts within your code as well as contracts with code outside of your project."
From https://www.typescriptlang.org/docs/handbook/interfaces.html

In brief, normal javascript is very *very* flexible when it comes to defining variables and their types
Typescript allows us to add more rigor and structure to that system so that we can define complicated
variables and structures that must satisfy certain conditions.
This is very powerful when you are likely to be creating many instances of the same type of variable
(a process which is easily prone to error) where errors can cause the code to faile (and be time consuming
to debug)

For example, consider the first interface we define QuestionnaireText and the first instance of its use

----
interface QuestionnaireText {
    type: string;
    label: string;
    id: string;
    name: string;
    title?: string;
}

var FirstName: QuestionnaireText = {
    type: QuestionnaireKeys.text,
    label: 'First Name:',
    id: 'first-name',
    name: 'firstname',
    title: 'Enter your name!'
};
----

Here, we are saying that any variable of type QuestionnaireText must have the following properties
    type
    label
    id
    name
Additionally it has the optional property
    title

Whenever we create a variable which we give the type QuestionnaireText, if we don't include some of the
mandatory properties or give them the wrong value type i.e. try to give label a boolean value or integer value
the compiler will flag it as an error
Introducing this kind of strict typing prevents us from making mistakes that would cause our web task
to function incorrectly

This is a very powerful system, giving us a lot of flexibility and capabilities.
For example, consider the QuestionnaireRadioset

----
interface QuestionnaireRadioSet {
    type: string;
    elements: QuestionnaireRadio[];
    title?: string;
}
----

Here we say that any variable of type QuestionnaireRadioSet must contain a property elements which is itself
an array of type QuestionnaireRadio i.e. an array of objects of type QuestionnaireRadio.

By creating these kind of interfaces, we can quickly make sure we are always adding the right properties to
objects when we create them, preventing some of the more common coding mistakes from occuring.

-------------
**TEMPLATES**
-------------

In the templates, we introduce handlebars expressions and handlebars helpers.

Going back to the code, consider our first call to gorilla.populate

gorilla.populate('#' + elementID, 'questionnaireText', Questionnaire[i]);

This is the same form as we've seen previously, except now we are using that third arguement to pass in an object
to gorilla.populate.

In the case of our first questionnaire element, this object will be 
{
    type: QuestionnaireKeys.text,
    label: 'First Name:',
    id: 'first-name',
    name: 'firstname',
    title: 'Enter your name!'
}

Compare this to our template "questionnaireText"

{{#if title}}<p><strong>{{title}}</strong></p>{{/if}}
<label for="{{id}}">{{label}}</label>
<input type="text" id="{{id}}" name="{{name}}">

The sections in double curly braces "{{title}}" are handlebars expressions.
"This expression means "look up the title property in the current context""
From http://handlebarsjs.com/expressions.html

In this case the "context" is the object that we have just passed into gorilla.populate.
In the place of {{id}} we will put the value of id from the object.
In the place of {{name}} we will put the value of name from the object.

Using this, we've created a very simple, barebones template that we can reuse again and again to create
a variety of different content!

This template also demonstrates a handlebars helper if.
These helpers are actually block helpers which have some special behaviour when it comes to context
Learn more about block helpers: http://handlebarsjs.com/builtin_helpers.html

In this case we are saying if the current context has a value for the key title, we're going to do something
This ties in with title being an optional property in our interface.
If we don't include this property, then this section of the template will not display.

Another handlebars helper is each
{{#each elements}}
    **create some content**
{{/each}}

Here we're saying that for each object in elements create some content.
There's another important concept that comes in here, which is context.
Within the each statement, the context is the contents of the current elements object.
We know from our interface that elements is an array of QuestionnaireRadio objects.
So, the context for this section of the template will match the properties we know are found
in a QuestionnaireRadio object, which are 
{
    label: string;
    id: string;
    name: string;
    value: string;
}

Context can be quite a tricky subject when you first get started.
If you find that content isn't displaying as you'd expect when using #each or other block helpers 
check that you've understood the context these helpers are working with.
It may be they're trying to access properties which aren't available to them.


*/

readme-structure file

/* README STRUCTURE
This file contents a simple overview of how coding tasks in Gorilla are constructed compared to the 
common structure of web applications which may be faimilar to you.
If you want to know more about the specfics of this tasks implementation, look at the file 'README'

--------
**CODE**
--------

In these files, we build the functionality and interactivity of our web task.
Here you place code that you may normally wrap in <script> tags.

Note that you don't need to wrap any of this code in <script> tags - Gorilla does that bit for you.

Its in these files that most of the code unique to Gorilla will appear.
That said, that's likely to be only a couple of lines - most of your web task will standard web application code.

In these files you can use Typescript (a typed superset of Javascript - most of the time, if you type normal 
Javascript, it will work), and jQuery.  
In brief, typescript allows us to create the functionality of our web application and jQuery allows 
us to link that functionality to elements on our page (our 'templates')

To learn more about Typescript: https://www.typescriptlang.org/docs/home.html
To learn more about jQuery: http://learn.jquery.com/

*********
At the top of these code files, you'll begin by including any other files that you may need.
These are other code files that you have written in this project, or files provided by Gorilla, 
like 'gorilla' and 'stateMachine'.
These **aren't** any files you may have uploaded to Resources or links you've included in the Head -
Gorilla will automatically include these for you, though you may need to add some additional code for
the editor to recognise the typing.

You'll probably need to put most of your code inside of gorilla.ready

gorilla.ready(()=>{
    ** Your code goes here **
})

This is our wrapper around the standard $(document).ready() function, used to determine when the page 
is ready for scripts to run and function correctly.
If you have any code that doesn't interact with or need to be bound to any elements on your page - like
preparing arrays of variables or helper functions, these can go outside of gorilla.ready

You'll need to use gorilla.populate to add content to your screen

gorilla.populate(**jQuery selector for element to add content to**, **name of template**, **object of variables**);

Most of the time, you'll want to add content to the main div we give you on the page, which has the id gorilla
and you probably won't need to add any variables to the page

Finally, you'll need to use gorilla.metric to upload any data you want collected to the Gorilla database

gorilla.metric(
    **object of key-value pairs**
)

In the 'Metrics' tab under the header 'Experiment', you set up the keys that you want Gorilla to store - these 
keys will need to match the keys you use in gorilla.metric.
A number of basic metrics are given to you by default, such as created time, participant ID.
Look at your download to see!

-------------
**TEMPLATES**
-------------

In these files you'll structure the content of your web application.
The names of these templates will likely be used in your gorilla.populate call
These templates contain content that you would normally find inside the <body> tags of a HTML webpage.
Note that you don't need to place <body> tags anywhere, nor do you need to include the commonly seen <HTML> tag
that appears at the very top of a webpage.

These files generally include no content unique to Gorilla - any standard HTML5 code will work in these files.
You can add classes and id's to elements as normal and add styling to them directly.

These templates are actually .handlebars files, so as well as standard HTML5 you can also use all the richness 
afforded by Handlebars and it's dynamic web content capabilities.

To learn more about HTML5: https://developer.mozilla.org/en-US/docs/Web/Reference
To learn more about Handlebars: http://handlebarsjs.com/

---------
**STYLE**
---------
USER WARNING - Style files (css) can take a long time to compile.  When you're adding new styling, make sure
the indicator at the bottom of the page has fully cycled through compiling and turned green.
Otherwise, your styling may not appear on your page.

In these files, you'll add any css styling that you want to attach to certain elements, classes or ids 
in your templates.
This file works exactly like a normal css style file would.
Don't forget to add the line

@import '/style/style.less';

To include the basic styling of Gorilla!

To learn more about CSS: https://developer.mozilla.org/en-US/docs/Learn/CSS/Introduction_to_CSS

*/

main file

// This task demonstrates how to use Typescript and Handlebars templates to custom build a Questionnaire using
// basic HTML 5 form elements
// It will include much the same functionality as the 'BasicQuestionnaireUsingForms' code task, except that,
// rather than using a premade questionnaire in a template, we'll dynamically build the questionnaire
// from smaller elements

// To do this, we'll demonstrate how to use Typescript interfaces, to make sure you're always including
// all the data necessary in an object
// We'll also demonstrate how to make use of Handlebars variables and Handlebar Helpers such as if and each

// For the sake of simplicity and brevity, we'll only demonstrate a couple of form elements (text and radio)

// If you want to know more about the different HTML Form elements take a look at these online guides
// https://www.w3schools.com/html/html_forms.asp
// https://developer.mozilla.org/en-US/docs/Learn/HTML/Forms

// To learn more about Typescript interfaces: https://www.typescriptlang.org/docs/handbook/interfaces.html
// To learn more about Handlebars: http://handlebarsjs.com/expressions.html
// ------------------------------------------------------------------------------

// import the basic gorilla functionality
import gorilla = require("gorilla/gorilla");

// import the additional files we've created and gain access to their functions and public contents
import questionnaire = require('questionnaire');
import basicQuestionnaire = require('basicQuestionnaire');

// this is our wrapper around $(document).ready
// this function is called when the page is ready
gorilla.ready(()=>{
   
    // populate the div with id 'gorilla' with the contents of the template 'basicForm', with no additional variables
    // a div with id 'gorilla' is the default that you recieve on every page
    gorilla.populate('#gorilla', 'basicForm', {});
    
    // Now we want to build our basic questionnaire
    // like in basicQuestionnaire, we use the dot notation to gain access to another files contents and functions
    var firstQuestionnaire = basicQuestionnaire.CreateBasicQuestionnaire();
    
    // Now that we have our basic form in place, and have retrieved the contents to go in our questionnaire
    // we're going to start adding our questionnaire elements to the display.
    for(var i = 0; i < firstQuestionnaire.length; i++){
        // we begin by adding a new div inside our dynamic-form div
        var elementID = 'element-' + i;
        $('.dynamic-form').append('<div id="' + elementID + '"></div></br>');
        
        // now depending on what our Questionnaire element is, we need to populate this div with different content
        switch(firstQuestionnaire[i].type){
            case questionnaire.QuestionnaireKeys.text:
                gorilla.populate('#' + elementID, 'questionnaireText', firstQuestionnaire[i]);
                break;
                
            case questionnaire.QuestionnaireKeys.radioset:
                gorilla.populate('#' + elementID, 'questionnaireRadioset', firstQuestionnaire[i]);
                break;
        }
    }
   
    // Now we need to bind our submit button 
    $('#myForm').submit(function(event){
        // First, we need to prevent the default action that usually occurs on a form submit button
        // we want to handle collation and submission of data ourselves
        event.preventDefault();
        // Collate all of our form elements
        var x = document.getElementsByTagName("input");
        for (var i = 0; i < x.length; i++) {
            
            // This is quite a simple implementation
            // It will upload the data from every single element
            // For instance, with radio buttons and checkboxes, you probably only want the checked elements
            // Using x[i].type to see if its a radio or checkbox, you can do some filtering to remove the unchecked elements
            var name = x[i].name;
            var value = x[i].value;
            var checked = (x[i].checked ? '1' : '');
            
            // upload the result to the metrics
            // gorilla.metric takes as an argument an object of key-value pairs
            // In the 'Metrics' tools, we've set three keys: name, value and checked
            // We've also given them titles for our spreadsheet
            gorilla.metric({
                name: name,
                value: value,
                checked: checked,
            });
                
        }
        
        // gorilla.finish() ends the current task and moves on to the next node in the experiment tree
        gorilla.finish();
    });
});

questionnaire file

// In this file, we're going to define all the basic elements that we could include in our questionnaire
// We're going to do this using Typescript interfaces
// To learn more about Typescript interfaces: https://www.typescriptlang.org/docs/handbook/interfaces.html
// Also, see the file 'README' for a description of how interfaces are used in this code

// First, we're going to create an object of key-value pairs which we'll use as identifiers for our
// Questionnaire elements
// Creating this will make assigning these identifies quick and less prone to error
// We're going to export this so its available to other files

export var QuestionnaireKeys = {
    text: 'text',
    radioset: 'radioset',
};

// Now we're going to create a set of interfaces which we will also export so they will be available in other files
// These interfaces allow us to create variables that have defined structures and contents
// This makes it less likely that we'll make an error when creating a complex object

// For example, with the definition below, when we create a variable of the type QuestionnaireText, 
// we now know that it must contain the properties type, label, id and name, all of which are strings
// If we don't give it one of these, the compiler will remind us that it needs to be there
// We also indicate (using a question mark) that it contains an optional variable title
// it won't complain if we don't give it this variable
export interface QuestionnaireText {
    type: string;
    label: string;
    id: string;
    name: string;
    title?: string;
};

// Radio buttons on a questionnaire usually appear in a set
// So we're going to define an interface whose only property is an array of type QuestionnaireRadio
export interface QuestionnaireRadioSet {
    type: string;
    elements: QuestionnaireRadio[];
    title?: string;
};

// In QuestionnaireRadio, we define the properties we'll need to create our radio buttons
export interface QuestionnaireRadio {
    label: string;
    id: string;
    name: string;
    value: string;
};

basicquestionniare file// In this file, we're going to use the interfaces we created previously to create a basic questionnaire
// This will just consist of a few simple text entry boxes and radio elements

// To being, import the contents of questionnaire, so we can gain access to the questionnaire keys
// and also the interfaces which will make sure that we create the objects we need correctly
import questionnaire = require('questionnaire');

// Define an array to hold our questionnaire elements
export var Questionnaire: any = [];

// Now we export a function which we'll use to build our array of questionnaire objects
// It will return this completed array at the end
export function CreateBasicQuestionnaire(){

    // we use the dot notation to gain access to the contents of questionnaire
    // Here, we want to define an object of type QuestionnaireText
    // so we use questionnaire.QuestionnaireText to access this interface from the questionnaire file
    var FirstName: questionnaire.QuestionnaireText = {
        type: questionnaire.QuestionnaireKeys.text,
        label: 'First Name:',
        id: 'first-name',
        name: 'firstname',
        title: 'Enter your name!'
    };
    
    Questionnaire.push(FirstName);
    
    var LastName: questionnaire.QuestionnaireText = {
        type: questionnaire.QuestionnaireKeys.text,
        label: 'Last Name:',
        id: 'last-name',
        name: 'lastname'
    };
    
    Questionnaire.push(LastName);
    
    // Create our radio button set
    var Handedness: questionnaire.QuestionnaireRadioSet = {
        type: questionnaire.QuestionnaireKeys.radioset,
        elements: [],
        title: 'Left handed, right handed or ambidextrous?',
    }
    
    var HandednessLeft: questionnaire.QuestionnaireRadio = {
        label: 'Left',
        id: 'handedness-left',
        name: 'handedness',
        value: 'left',
    }
    
    Handedness.elements.push(HandednessLeft);
    
    var HandednessRight: questionnaire.QuestionnaireRadio = {
        label: 'Right',
        id: 'handedness-right',
        name: 'handedness',
        value: 'right',
    }
    
    Handedness.elements.push(HandednessRight);
    
    var HandednessAmbi: questionnaire.QuestionnaireRadio = {
        label: 'Ambidextrous',
        id: 'handedness-ambi',
        name: 'handedness',
        value: 'ambi',
    }
    
    Handedness.elements.push(HandednessAmbi);
    
    Questionnaire.push(Handedness);
    
    return Questionnaire;
}

radiosetup file
{{#if title}}<p><strong>{{title}}</strong></p>{{/if}}
{{#each elements}}
    <input type="radio" id="{{id}}" name="{{name}}" value="{{value}}" >
    <label for="{{id}}">{{label}}</label>
{{/each}}

questionnairetext file
<!-- Into this template, we'll be passing an object of type QuestionnaireText
This will contain an id, a label and a name (we don't need to use the type here
It may also contain a title element
We use the Handlebars Helper if to control whether this element displays
-->
{{#if title}}<p><strong>{{title}}</strong></p>{{/if}}
<label for="{{id}}">{{label}}</label>
<input type="text" id="{{id}}" name="{{name}}">

basicform file
<div class="container">
    <form id="myForm">
        <h1>Enter some information!</h1>
        <div class="dynamic-form">
            
        </div>
        <input type="submit" style="margin-top: 10px;">
    </form>
</div> 

style file
@import '/style/style.less';

metrics.md file (Add metrics, which detail which values from gorilla.metric() you want to come through in your data CSVs, and allow you to specify the column names.)
name
value
checked
*Documentation compiled from Gorilla official support pages. Last updated: 2025-10-13*
