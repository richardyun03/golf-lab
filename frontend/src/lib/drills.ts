export type Difficulty = "beginner" | "intermediate" | "advanced";

export interface Drill {
  id: string;
  name: string;
  difficulty: Difficulty;
  duration: string;
  equipment: string[];
  steps: string[];
  why_it_works: string;
  reps: string;
}

export interface FaultTrainingPlan {
  fault_type: string;
  title: string;
  summary: string;
  key_feels: string[];
  drills: Drill[];
}

export const FAULT_TRAINING: Record<string, FaultTrainingPlan> = {
  sway: {
    fault_type: "sway",
    title: "Lateral Sway",
    summary:
      "Your hips are sliding laterally away from the target during the backswing instead of rotating. This makes it hard to return to the ball consistently and costs you power.",
    key_feels: [
      "Weight stays on the inside of your trail foot",
      "Trail hip rotates behind you, not sideways",
      "Imagine your spine as a fixed axis — hips turn around it",
    ],
    drills: [
      {
        id: "sway-wall",
        name: "Wall Drill",
        difficulty: "beginner",
        duration: "5 min",
        equipment: ["Wall or door frame"],
        steps: [
          "Set up in your golf posture with your trail hip touching a wall.",
          "Make slow backswings keeping your trail hip in contact with the wall.",
          "If your hip pushes into the wall or loses contact, you're swaying.",
          "Focus on feeling the hip rotate rather than slide.",
        ],
        why_it_works:
          "The wall provides instant tactile feedback — any lateral movement is immediately felt. Trains your body to distinguish rotation from translation.",
        reps: "3 sets of 10 slow swings",
      },
      {
        id: "sway-alignment-stick",
        name: "Alignment Stick Gate",
        difficulty: "intermediate",
        duration: "10 min",
        equipment: ["Alignment stick", "Golf club"],
        steps: [
          "Stick an alignment rod in the ground just outside your trail hip at address.",
          "Make full backswings — your hip should not push past the stick.",
          "Start with half swings and gradually increase to full speed.",
          "Hit balls once you can consistently stay inside the gate.",
        ],
        why_it_works:
          "Creates a physical boundary that your body learns to respect. Transfers directly to on-course feel because you're hitting real shots.",
        reps: "20 balls, then remove the stick and hit 10 more",
      },
      {
        id: "sway-single-leg",
        name: "Trail Leg Balance Swings",
        difficulty: "advanced",
        duration: "10 min",
        equipment: ["Golf club", "Range balls"],
        steps: [
          "Address the ball normally, then lift your lead foot slightly off the ground.",
          "Make three-quarter backswings balancing on your trail leg.",
          "If you sway, you'll lose balance immediately.",
          "Once comfortable, put both feet down and hit full shots carrying the feel.",
        ],
        why_it_works:
          "Eliminates sway entirely — any lateral movement topples you. Forces your body to find a rotational solution. Builds proprioception for the correct weight load.",
        reps: "10 balance swings, then 15 full shots",
      },
    ],
  },

  slide: {
    fault_type: "slide",
    title: "Lateral Slide",
    summary:
      "Your hips are sliding toward the target in the downswing instead of rotating open. This leads to blocks, pushes, and inconsistent low point.",
    key_feels: [
      "Lead hip turns behind you, not sideways",
      "Feel like your belt buckle faces the target at finish, not your hip bones",
      "Pressure shifts to lead foot but the hip stays centered",
    ],
    drills: [
      {
        id: "slide-chair",
        name: "Chair Bump Drill",
        difficulty: "beginner",
        duration: "5 min",
        equipment: ["Chair or stool"],
        steps: [
          "Place a chair just outside your lead hip at address.",
          "Your hip should lightly touch the chair — not push it.",
          "Make downswings focusing on rotating open rather than bumping the chair.",
          "The chair should barely move if you're rotating correctly.",
        ],
        why_it_works:
          "Gives you a physical limit for lateral movement. If the chair moves more than an inch, you're sliding.",
        reps: "3 sets of 10 swings",
      },
      {
        id: "slide-step-drill",
        name: "Step-Through Drill",
        difficulty: "intermediate",
        duration: "10 min",
        equipment: ["Golf club", "Range balls"],
        steps: [
          "Take your normal backswing.",
          "Start the downswing, then step your lead foot toward the target (like a baseball swing).",
          "Feel how the step creates rotation rather than slide.",
          "Gradually reduce the step size until your feet stay planted but the rotation feel remains.",
        ],
        why_it_works:
          "Exaggerates the rotational sequence. The step forces your body to rotate rather than slide because your base is actively moving.",
        reps: "15 step-through shots, then 15 normal shots",
      },
      {
        id: "slide-resistance-band",
        name: "Resistance Band Rotation",
        difficulty: "advanced",
        duration: "10 min",
        equipment: ["Resistance band", "Anchor point (fence post, bag stand)"],
        steps: [
          "Loop a resistance band around your lead hip and anchor it to your lead side.",
          "The band should pull you toward the target — resist it by rotating, not bracing laterally.",
          "Make slow-motion downswings feeling the hip rotate against the band tension.",
          "Progress to full-speed swings with the band.",
        ],
        why_it_works:
          "The band exaggerates the slide tendency. Learning to resist it with rotation rewires your motor pattern under load.",
        reps: "3 sets of 8 swings, increasing speed",
      },
    ],
  },

  early_extension: {
    fault_type: "early_extension",
    title: "Early Extension",
    summary:
      "You're losing your spine angle through impact — hips are thrusting toward the ball. This causes thin shots, blocks, and hooks as your body runs out of room.",
    key_feels: [
      "Maintain your tush line — glutes stay back through impact",
      "Feel like you're sitting into the shot",
      "Your belt buckle rotates toward the target while your hips stay back",
    ],
    drills: [
      {
        id: "ee-tush-line",
        name: "Tush Line Drill",
        difficulty: "beginner",
        duration: "5 min",
        equipment: ["Alignment stick or chair"],
        steps: [
          "Place an alignment stick vertically behind you so your glutes just touch it at address.",
          "Make slow swings maintaining contact with the stick through impact.",
          "If your glutes lose contact before impact, you're extending early.",
          "Start with half swings and gradually build up.",
        ],
        why_it_works:
          "Directly targets the tush line — the single best indicator of early extension. The stick gives instant feedback on whether your pelvis is drifting forward.",
        reps: "20 slow swings, 10 at full speed",
      },
      {
        id: "ee-squat-turn",
        name: "Squat-to-Turn Drill",
        difficulty: "intermediate",
        duration: "10 min",
        equipment: ["Golf club"],
        steps: [
          "From the top, feel like you squat slightly (drop 1-2 inches) before rotating.",
          "This squat keeps your pelvis from thrusting forward.",
          "Exaggerate the feel of sitting down and then spinning through.",
          "Hit balls focusing on the squat feel at transition.",
        ],
        why_it_works:
          "The slight squat move pre-loads the legs and makes it physically difficult to thrust forward. Many tour pros have this move — it's a feature, not a drill artifact.",
        reps: "20 balls with exaggerated squat, 10 normal",
      },
      {
        id: "ee-cross-arm",
        name: "Cross-Arm Rotation Drill",
        difficulty: "beginner",
        duration: "5 min",
        equipment: [],
        steps: [
          "Cross your arms over your chest (no club).",
          "Get into golf posture and rotate to a backswing position.",
          "Now rotate through to a finish while keeping your spine angle constant.",
          "Watch yourself in a mirror — your head should stay at the same height.",
        ],
        why_it_works:
          "Removes the club to isolate the body rotation. Without a club to hit at, your body naturally finds the correct rotational pattern. Mirror feedback makes the fault visible.",
        reps: "3 sets of 15 rotations",
      },
    ],
  },

  chicken_wing: {
    fault_type: "chicken_wing",
    title: "Chicken Wing",
    summary:
      "Your lead arm is collapsing at the elbow through follow-through instead of extending. This reduces power, creates inconsistent contact, and often leads to slices.",
    key_feels: [
      "Both arms extend fully through the ball",
      "Feel like you're pushing the grip toward the target post-impact",
      "Lead arm stays connected to your chest rotation",
    ],
    drills: [
      {
        id: "cw-towel",
        name: "Towel Under Arm Drill",
        difficulty: "beginner",
        duration: "10 min",
        equipment: ["Small towel", "Golf club", "Range balls"],
        steps: [
          "Tuck a small towel under your lead armpit.",
          "Hit shots keeping the towel in place through follow-through.",
          "If the towel drops before your follow-through, your arm is disconnecting.",
          "Focus on rotating your body to keep the arm connected, not squeezing.",
        ],
        why_it_works:
          "Forces the lead arm to stay connected to the body's rotation. A chicken wing happens when the arm works independently — this prevents that.",
        reps: "20 balls with towel, 10 without",
      },
      {
        id: "cw-punch-shots",
        name: "Punch Shot Extension",
        difficulty: "intermediate",
        duration: "15 min",
        equipment: ["7 iron", "Range balls"],
        steps: [
          "Hit low punch shots with a three-quarter swing.",
          "Hold your finish with both arms extended toward the target.",
          "The ball should fly low and straight — a chicken wing produces a high weak fade.",
          "Gradually increase swing length while maintaining the extension feel.",
        ],
        why_it_works:
          "Punch shots require arm extension for proper compression. The shortened swing makes it easier to focus on the extension without full-swing complications.",
        reps: "30 punch shots",
      },
      {
        id: "cw-split-grip",
        name: "Split-Grip Swings",
        difficulty: "advanced",
        duration: "10 min",
        equipment: ["Golf club"],
        steps: [
          "Grip the club with a 2-inch gap between your hands.",
          "Make half swings — the split grip makes chicken wing physically uncomfortable.",
          "Focus on both arms extending through impact.",
          "When comfortable, close the gap and hit full shots.",
        ],
        why_it_works:
          "The split grip mechanically prevents the lead elbow from collapsing because both hands need to work together through the hitting zone.",
        reps: "15 split-grip, 15 normal grip",
      },
    ],
  },

  casting: {
    fault_type: "casting",
    title: "Casting / Early Release",
    summary:
      "You're releasing your wrist angle too early in the downswing. This throws away lag and clubhead speed — like casting a fishing rod instead of cracking a whip.",
    key_feels: [
      "Butt of the club points at the ball until hands pass your trail thigh",
      "Feel like you're pulling the grip down, not swinging the clubhead",
      "Wrists stay cocked until your body rotation unwinds them naturally",
    ],
    drills: [
      {
        id: "cast-pump",
        name: "Pump Drill",
        difficulty: "beginner",
        duration: "10 min",
        equipment: ["Golf club"],
        steps: [
          "Take the club to the top of your backswing.",
          "Bring it halfway down (hands at hip height) — check that wrists are still fully hinged.",
          "Pump back up to the top.",
          "Repeat 3 times, then swing through on the fourth.",
        ],
        why_it_works:
          "The pumps train the feel of maintaining wrist hinge in the early downswing. By the time you swing through, the pattern is loaded into your muscle memory for that rep.",
        reps: "10 sets of pump-pump-pump-swing",
      },
      {
        id: "cast-lag-towel",
        name: "Lag Towel Whip",
        difficulty: "intermediate",
        duration: "10 min",
        equipment: ["Golf towel"],
        steps: [
          "Hold a golf towel by one end like a golf club.",
          "Make a full backswing and downswing — the towel should snap at the bottom.",
          "If you cast, the towel flops limply with no snap.",
          "Focus on keeping your hands ahead and letting the towel whip through naturally.",
        ],
        why_it_works:
          "A towel has no weight to manipulate — the only way to make it snap is with proper sequencing and lag. Teaches the correct timing without ball-hitting anxiety.",
        reps: "30 towel whips",
      },
      {
        id: "cast-right-arm",
        name: "Trail Arm Only Swings",
        difficulty: "advanced",
        duration: "15 min",
        equipment: ["Short iron", "Range balls"],
        steps: [
          "Grip a short iron with only your trail hand.",
          "Make smooth half swings focusing on keeping the wrist angle until impact.",
          "The trail arm naturally maintains lag when it's doing all the work.",
          "Gradually add the lead hand back, carrying the feel.",
        ],
        why_it_works:
          "Isolates the trail arm's role in maintaining lag. Most casting is caused by the lead arm pulling the club out of lag — removing it solves the problem temporarily so you can learn the feel.",
        reps: "15 trail-arm only, 15 both hands",
      },
    ],
  },

  over_the_top: {
    fault_type: "over_the_top",
    title: "Over the Top",
    summary:
      "Your hands are moving outward (away from your body) in the early downswing, creating an outside-in swing path. This is the most common cause of slices and pull shots.",
    key_feels: [
      "Feel like your hands drop straight down from the top",
      "Trail elbow slots into your trail hip pocket",
      "The club approaches the ball from the inside, not over the top",
    ],
    drills: [
      {
        id: "ott-headcover",
        name: "Headcover Path Drill",
        difficulty: "beginner",
        duration: "15 min",
        equipment: ["Driver headcover", "Golf club", "Range balls"],
        steps: [
          "Place a headcover about 6 inches behind the ball and 2 inches outside the target line.",
          "Hit shots without touching the headcover.",
          "An over-the-top path will hit the headcover; an inside path avoids it.",
          "Start with half swings and work up to full speed.",
        ],
        why_it_works:
          "Provides a visual and physical consequence for an outside path. Your brain will quickly learn to route the club inside to avoid the obstacle.",
        reps: "25 balls",
      },
      {
        id: "ott-trail-elbow",
        name: "Trail Elbow Pocket Drill",
        difficulty: "intermediate",
        duration: "10 min",
        equipment: ["Golf towel", "Golf club"],
        steps: [
          "Tuck a small towel under your trail armpit.",
          "Make downswings keeping the towel in place — this forces your trail elbow to slot against your body.",
          "An OTT move throws the elbow out and drops the towel.",
          "Hit balls with the towel, focusing on the elbow staying close.",
        ],
        why_it_works:
          "The trail elbow separating from the body is the mechanical cause of OTT. Keeping it tucked physically prevents the outside move.",
        reps: "20 balls with towel, 10 without",
      },
      {
        id: "ott-close-stance",
        name: "Closed Stance Draws",
        difficulty: "advanced",
        duration: "15 min",
        equipment: ["7 iron", "Range balls"],
        steps: [
          "Align your feet 20-30 degrees closed (aiming right of target).",
          "Hit shots trying to start the ball along your foot line.",
          "This setup makes OTT nearly impossible — you'll hit big hooks if you come over it.",
          "Gradually open your stance back to normal, carrying the inside-out feel.",
        ],
        why_it_works:
          "A closed stance creates a physical environment where only an inside path produces a playable shot. Trains the correct motor pattern under exaggerated conditions.",
        reps: "20 closed stance, 10 normal stance",
      },
    ],
  },

  reverse_pivot: {
    fault_type: "reverse_pivot",
    title: "Reverse Pivot",
    summary:
      "Your weight is shifting toward the target during the backswing instead of loading onto your trail side. This robs you of power and often causes fat/thin contact.",
    key_feels: [
      "Feel your weight load into the inside of your trail foot on the backswing",
      "Lead shoulder works down toward the ball, not up and away",
      "At the top, you should feel grounded on your trail side",
    ],
    drills: [
      {
        id: "rp-step-back",
        name: "Step-Back Drill",
        difficulty: "beginner",
        duration: "5 min",
        equipment: ["Golf club"],
        steps: [
          "Start with your feet together at address.",
          "As you begin the backswing, step your trail foot back into position.",
          "This forces your weight onto the trail side.",
          "Make full swings with this step-back feeling.",
        ],
        why_it_works:
          "The stepping motion physically loads weight onto the trail side. It's impossible to reverse pivot when your trail foot is actively moving back.",
        reps: "20 step-back swings",
      },
      {
        id: "rp-lift-lead",
        name: "Lead Foot Lift Test",
        difficulty: "intermediate",
        duration: "10 min",
        equipment: ["Golf club", "Range balls"],
        steps: [
          "Make a backswing and pause at the top.",
          "Try to lift your lead foot off the ground.",
          "If you can't balance, your weight is on the wrong side (reverse pivot).",
          "Practice until you can comfortably lift the lead foot at the top, then hit balls.",
        ],
        why_it_works:
          "Simple binary test — if you can't lift your lead foot, you're reverse pivoting. Gives you a clear success/fail checkpoint at the top of every swing.",
        reps: "10 balance checks, then 20 balls",
      },
      {
        id: "rp-downhill",
        name: "Downhill Lie Practice",
        difficulty: "advanced",
        duration: "15 min",
        equipment: ["Golf club", "Range balls", "Slight downhill slope"],
        steps: [
          "Find a slight downhill lie (trail foot higher than lead foot).",
          "Hit balls from this lie — the slope naturally loads weight onto your trail side.",
          "Feel how the slope makes your lead shoulder work downward.",
          "Return to flat ground and replicate the feel.",
        ],
        why_it_works:
          "The slope does the work for you — gravity pulls your weight into the correct position. Once you feel the difference, you can replicate it on flat ground.",
        reps: "20 downhill shots, 10 on flat ground",
      },
    ],
  },

  excessive_head_movement: {
    fault_type: "excessive_head_movement",
    title: "Excessive Head Movement",
    summary:
      "Your head is moving too much from address through impact — either sliding laterally or dipping vertically. A steady head is the foundation of consistent contact.",
    key_feels: [
      "Your head is the top of a pendulum — it stays still while everything swings below",
      "Eyes stay fixed on the ball; head rotates slightly but doesn't translate",
      "Feel tall through the swing — no dipping",
    ],
    drills: [
      {
        id: "hm-shadow",
        name: "Shadow Drill",
        difficulty: "beginner",
        duration: "5 min",
        equipment: ["Sunny day or bright light"],
        steps: [
          "Stand so your shadow is clearly visible on the ground.",
          "Make swings watching the shadow of your head.",
          "The shadow should stay in roughly the same spot from address to impact.",
          "Any large movement is immediately visible.",
        ],
        why_it_works:
          "Your shadow acts as a real-time motion tracker. It's the simplest feedback loop — no equipment, instant visual confirmation.",
        reps: "15 swings watching your shadow",
      },
      {
        id: "hm-doorframe",
        name: "Doorframe Height Check",
        difficulty: "intermediate",
        duration: "10 min",
        equipment: ["Doorframe or low ceiling area", "Golf club"],
        steps: [
          "Stand in a doorframe so the top of your head nearly touches the frame at address.",
          "Make slow swings maintaining the same head height throughout.",
          "If you dip, your head drops away from the frame. If you rise, you bump it.",
          "Trains vertical stability specifically.",
        ],
        why_it_works:
          "Constrains vertical head movement with a physical reference. The slight claustrophobia of the doorframe also trains you to stay compact.",
        reps: "3 sets of 10 slow swings",
      },
      {
        id: "hm-partner-hold",
        name: "Partner Head Hold",
        difficulty: "beginner",
        duration: "10 min",
        equipment: ["Practice partner", "Golf club"],
        steps: [
          "Have a partner gently place their hand on top of your head at address.",
          "Make slow swings maintaining light pressure against their hand.",
          "Your partner can tell you exactly how and when your head moves.",
          "Graduate to full swings once you can stay stable in slow motion.",
        ],
        why_it_works:
          "Human feedback is richer than any drill aid. Your partner can describe exactly what's happening (dip at transition, slide on backswing, etc.) and you can feel the correction in real time.",
        reps: "20 slow swings, 10 at medium speed",
      },
    ],
  },
};

export function getTrainingPlan(faultType: string): FaultTrainingPlan | null {
  return FAULT_TRAINING[faultType] ?? null;
}

export function getAllFaultTypes(): string[] {
  return Object.keys(FAULT_TRAINING);
}
